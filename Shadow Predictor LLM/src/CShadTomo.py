import qiskit
from qiskit.quantum_info import Pauli, Clifford, random_clifford, SparsePauliOp
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

import numpy as np
import time
import math
import torch

import matplotlib.pyplot as plt
from collections import Counter

from pathlib import Path
import json
from typing import Any, Dict, List, Optional, Tuple, Literal, Callable, Union

 
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# == Type settings ==
Predictor = Literal["classical", "llm"]
Scheme = Literal["local-pauli", "local-clifford", "global-clifford"]
OpsMode = Literal["save", "load", "predict_all"]

class CTomo:
    def __init__(self, n_qubits, scheme, num_snapshots, default_predictor, llm_client=None):
        self.n_qubits: int = n_qubits
        self.scheme: Scheme = scheme
        self.num_snapshots: int = num_snapshots

        self._global_inv_cache = {}

        self.seed: int = 42
        self.rng = np.random.default_rng(self.seed)

        self.default_predictor: Predictor = default_predictor
        self.llm_client = llm_client

        # LLM performance knobs
        self.llm_batch_size: int = 16          # prompts processed together
        self.llm_max_new_tokens: int = 24      # our JSON is tiny; keep short
        self.llm_use_fast_prompt: bool = True  # compact prompt (faster than chat)


        self.storage_path: str = str(Path.cwd() / "data" / "snapshots.json")

        self.snapshots: list[dict] = []
        self.observables: list[str] = []

        # Load 24×1q Clifford table once (strings "h","s","sdg")
        table_path = Path("data/clifford_1q_gates.json")
        self.clifford_table = json.loads(table_path.read_text()) if table_path.exists() else None

        self.backend = None

    # --- Config ---
    def set_backend(self, backend: Any) -> None:
        """Attach execution backend."""
        # Initialize your account - DO THIS LATER WHEN YOU AREN'T SIMULATING RESULTS
        self.backend = AerSimulator()
        

    def set_state_prep(self, fn: Callable[..., Any]) -> None:
        """Attach state preparation routine."""
        self.state_prep = fn
        

    def set_observables(self, observables: List[str]) -> None:
        """Register observables O_i to estimate."""
        for s in observables:
            if len(s) != self.n_qubits or any(c not in "IXYZ" for c in s):
                raise ValueError(f"Invalid Pauli string: {s}")
        self.observables.extend(observables)
        

    def set_llm(self, model_id: str) -> None:
        # Pick device: prefer MPS on Apple, then CUDA, else CPU
        device = (
            "mps" if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        use_fp16 = device in ("mps", "cuda")

        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token

        # IMPORTANT: no auto sharding; load whole model and move to device
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=(torch.float16 if use_fp16 else torch.float32),
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        model.to(device)
        model.eval()

        # Text-generation pipeline with batching, short outputs
        self.llm_client = pipeline(
            "text-generation",
            model=model,
            tokenizer=tok,
            return_full_text=False,
            do_sample=False,                  # greedy for JSON
            top_p=1.0,
            max_new_tokens=self.llm_max_new_tokens,  # from your knobs
            batch_size=self.llm_batch_size,          # from your knobs
            # NOTE: we don’t pass device_map here; pipeline will use model.device
        )

        self._llm_is_chat = hasattr(tok, "apply_chat_template")
        self._has_chat_template = bool(getattr(tok, "chat_template", None))
        self._tok = tok
        

    # --- Snapshot generation ---
    def sample_bases(self) -> dict:
        if self.scheme == "local-pauli":
            bases = self.rng.integers(0, 3, size=self.n_qubits)  # 0:Z, 1:X, 2:Y
            return {"scheme": "local-pauli", "bases": bases.tolist()}
    
        elif self.scheme == "local-clifford":
            cliff_idx = self.rng.integers(0, 24, size=self.n_qubits)  # 0..23
            return {"scheme": "local-clifford", "cliff_idx": cliff_idx.tolist()}
    
        elif self.scheme == "global-clifford":
            snapshot_seed = int(self.rng.integers(0, 2**32))
            return {"scheme": "global-clifford", "seed": snapshot_seed, "n": self.n_qubits}
    
        else:
            raise ValueError("Unknown scheme.")
        

    def apply_1q_clifford(self, qc: QuantumCircuit, qubit: int, cliff_idx: int):
        if self.clifford_table is None:
            raise RuntimeError("Missing data/clifford_1q_gates.json")
        for g in self.clifford_table[int(cliff_idx)]:
            getattr(qc, g)(qubit)
        
    
    # helper: conjugate Z by a sequence of 1q gates ("h","s","sdg")
    def _cz_effect(self, seq: list[str]) -> tuple[str, int]:
        axis, sign = "Z", +1
        for gate in seq:
            if gate == "h":
                if axis == "X": axis = "Z"
                elif axis == "Z": axis = "X"
                elif axis == "Y": sign, axis = -sign, "Y"
            elif gate == "s":
                if axis == "X": axis = "Y"
                elif axis == "Y": sign, axis = -sign, "X"
                # Z unchanged
            elif gate == "sdg":
                if axis == "X": sign, axis = -sign, "Y"
                elif axis == "Y": axis = "X"
                # Z unchanged
        return axis, sign  # axis in {"X","Y","Z"}, sign in {+1,-1}
    
    def build_snapshot_circuit(self, bases: dict) -> tuple[QuantumCircuit, dict]:
        # Build quantum circuit with n number of qubits
        qc = QuantumCircuit(self.n_qubits, self.n_qubits)

        # Initialise the state preparation
        self.state_prep(qc)
    
        axes = [0]*self.n_qubits     # 0:Z, 1:X, 2:Y  (effective measured axes)
        signs = [1]*self.n_qubits    # per-qubit overall sign from conjugation
    
        if bases["scheme"] == "local-pauli":
            # Map desired Pauli axes to basis-change gates
            for q, idx in enumerate(bases["bases"]):
                if idx == 1:       # X
                    qc.h(q)
                    axes[q], signs[q] = 1, +1
                elif idx == 2:     # Y
                    qc.sdg(q); qc.h(q)
                    axes[q], signs[q] = 2, +1
                else:              # Z
                    axes[q], signs[q] = 0, +1
    
        elif bases["scheme"] == "local-clifford":
            if self.clifford_table is None:
                # Fallback: synthesize the 1q Clifford and derive axis/sign directly
                for q in range(self.n_qubits):
                    U = random_clifford(1, seed=int(self.rng.integers(0, 2**32)))
                    qc.append(U.to_instruction(), [q])

                    # Conjugate Z by U in Heisenberg frame
                    Pp = Pauli("Z").evolve(U, frame="h")

                    # Sign from phase: 0→+1, 1→+i, 2→-1, 3→-i  (ignore imaginary; use ±1)
                    phase = Pp.phase % 4
                    sgn = +1 if phase in (0,) else -1

                    # Axis from x/z bits (for 1 qubit)
                    x = int(Pp.x[0]); z = int(Pp.z[0])
                    if x == 0 and z == 0:
                        axis_lbl = "I"
                    elif x == 1 and z == 0:
                        axis_lbl = "X"
                    elif x == 0 and z == 1:
                        axis_lbl = "Z"
                    else:
                        axis_lbl = "Y"

                    # If it becomes identity, treat as Z with +1 (no basis change)
                    if axis_lbl == "I":
                        axis_lbl, sgn = "Z", +1

                    axes[q]  = {"Z":0, "X":1, "Y":2}[axis_lbl]
                    signs[q] = sgn
            else:
                # Your existing table-driven path
                for q, idx in enumerate(bases["cliff_idx"]):
                    self.apply_1q_clifford(qc, q, int(idx))
                    axis_lbl, sgn = self._cz_effect(self.clifford_table[int(idx)])
                    axes[q] = {"Z":0, "X":1, "Y":2}[axis_lbl]
                    signs[q] = sgn
    
        elif bases["scheme"] == "global-clifford":
            C = random_clifford(self.n_qubits, seed=bases["seed"])
            qc.append(C.to_instruction(), range(self.n_qubits))
            axes, signs = None, None
    
        else:
            raise ValueError("Unknown scheme.")
    
        qc.measure(range(self.n_qubits), range(self.n_qubits))
        meta = {"scheme": bases["scheme"], "axes": axes, "signs": signs, **{k:v for k,v in bases.items() if k not in ("scheme",)}}
        return qc, meta
        



    
    def _inverse_one_snapshot(self, snapshot: dict, pauli_str: str) -> float:
        """
        Unified inverse:
          - Local schemes: use axes/signs/bits
          - Global Clifford: use precomputed or computed inverse
        """
        sch = snapshot.get("scheme")

        if sch in ("local-pauli", "local-clifford"):
            axes = snapshot.get("axes")
            signs = snapshot.get("signs")
            bits = snapshot["bits"]
            prod, m = 1.0, 0
            need_map = {"Z":0, "X":1, "Y":2}
            for q, p in enumerate(pauli_str):
                if p == "I": 
                    continue
                m += 1
                if axes[q] != need_map[p]:
                    return 0.0
                ev = +1.0 if bits[-1-q] == "0" else -1.0
                prod *= (signs[q] * ev)
            return (3.0**m) * prod
        elif sch == "global-clifford":
            return self._estimate_one_global_clifford(snapshot, pauli_str)


    def _estimate_one_global_clifford(self, snapshot: dict, pauli_str: str) -> float:
        """
        Global-Clifford inverse for a single Pauli-string observable.
        snapshot: {'scheme':'global-clifford','seed': int,'n': int,'bits': '...'}
        Implements:  ȳ_k = (d+1) * ⟨b | U O U† | b⟩  - Tr(O),  d=2^n
        For Pauli strings: Tr(O)=0 unless O = I...I, then Tr(O)=d.
        """
        n = snapshot["n"]
        seed = snapshot["seed"]
        bits = snapshot["bits"]
        d = 2 ** n
    
        # Recreate the Clifford U from the stored seed
        U = random_clifford(n, seed=seed)
    
        # Conjugate O by U: O' = U O U†  (use Pauli evolve in Heisenberg frame)
        P = Pauli(pauli_str)                   # single-term Pauli
        Pp = P.evolve(U, frame="h")            # conjugation
    
        # If O' has any X/Y on any qubit, ⟨b|O'|b⟩ = 0 in computational basis
        label = Pp.to_label()                  # e.g., 'IZXZ...'
        # Extract real phase: Pauli stores phase ∈ {0,1,2,3} ⇒ 1, i, -1, -i
        phase = Pp.phase % 4
        if phase in (1, 3):                    # ±i → expectation is purely imaginary ⇒ 0 for Hermitian O
            return 0.0
        sign = +1.0 if phase in (0,) else -1.0 # 0→+1, 2→-1
    
        if any(ch in ("X", "Y") for ch in label):
            ev = 0.0
        else:
            # Only I/Z remain: ⟨b|Z_j|b⟩ = (-1)^{b_j}
            ev = sign
            # Rightmost bit is qubit 0 in Qiskit bitstrings
            for q, ch in enumerate(label):
                if ch == "Z":
                    ev *= (+1.0 if bits[-1 - q] == "0" else -1.0)
    
        # Trace term
        trO = d if all(c == "I" for c in pauli_str) else 0.0
    
        return (d + 1) * ev - trO
    
    
    # --- Classical shadows path ---
    def estimate_observable_classical(self, O: Any) -> Tuple[float, float]:
        """Return (estimate, std_err) via classical shadows."""
        vals = np.empty(len(self.snapshots), dtype=float)
        w = 0
        for snap in self.snapshots:
            sch = snap["scheme"]
            if sch in ("local-pauli", "local-clifford"):
                vals[w] = self._inverse_one_snapshot(snap, O); w += 1
            elif sch == "global-clifford":
                vals[w] = self._estimate_one_global_clifford(snap, O); w += 1
            else:
                continue

        if w == 0:
            return 0.0, 0.0

        arr = vals[:w]
        mean = float(arr.mean())
        se = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
        return mean, se

        

    # ===================================== LLM METHODS =====================================

    # --- Helpers for LLM path ---
    def _encode_snapshot(self, snap: dict) -> dict:
        """Provider-agnostic snapshot encoding for the LLM."""
        out = {"scheme": snap["scheme"], "bits": snap["bits"]}

        # Prefer unified local encoding if it is present
        if "axes" in snap: out["axes"] = snap["axes"]
        if "signs" in snap: out["signs"] = snap["signs"]

        # Keep original descriptors for the LLM
        if "cliff_idx" in snap: out["cliff_idx"] = snap["cliff_idx"]
        if "seed" in snap: out["seed"] = snap["seed"]
        if "n" in snap: out["n"] = snap["n"]

        return out

    def _build_llm_prompt_fast(self, O: str, snap_enc: dict) -> str:
        """
        Ultra-compact single-string prompt for faster generation on M2/MPS.
        The model must output STRICT JSON only: {"value": float, "confidence": float}
        """
        payload = {"n": self.n_qubits, "O": O, "s": snap_enc}
        return (
            "Return STRICT JSON only:{\"value\":float,\"confidence\":float}. "
            "No extra text. Input:" + json.dumps(payload, separators=(",", ":"))
        )


    def _build_llm_prompt(self, O: str, snap_enc: dict) -> list[dict]:
        guide = (
            "You are estimating an expectation value from a single classical-shadow snapshot.\n"
            "Return STRICT JSON only: {\"value\": float, \"confidence\": float}\n"
            "No prose, no explanations. If unsure, use confidence near 0."
        )
        payload = {
            "n_qubits": self.n_qubits,
            "observable": O,
            "snapshot": snap_enc
        }
        user = (
            "Format: {\"value\": <float>, \"confidence\": <float>}.\n"
            "Input:\n" + json.dumps(payload, separators=(",", ":"))
        )
        # Chat messages
        return [
            {"role": "system", "content": guide},
            {"role": "user", "content": user},
        ]


    def _call_llm(self, messages_or_text) -> str:
        # Extract system and user content if messages are provided
        if isinstance(messages_or_text, list):
            sys_msg = next((m.get("content", "") for m in messages_or_text if m.get("role") == "system"), "")
            usr_msg = next((m.get("content", "") for m in messages_or_text if m.get("role") == "user"), "")
        else:
            sys_msg, usr_msg = "", str(messages_or_text)

        prompt = None
        if getattr(self, "_llm_is_chat", False) and getattr(self, "_tok", None):
            try:
                prompt = self._tok.apply_chat_template(
                    messages_or_text,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                prompt = None  # fall back below

        if prompt is None:
            # Try Llama-3 style manual template if tokens exist
            def _has_tok(tok):
                try:
                    return self._tok.convert_tokens_to_ids(tok) != self._tok.unk_token_id
                except Exception:
                    return False
            if getattr(self, "_tok", None) and all(_has_tok(t) for t in ("<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>")):
                BOS = "<|begin_of_text|>"; SH = "<|start_header_id|>"; EH = "<|end_header_id|>"; EOT = "<|eot_id|>"
                prompt = (
                    f"{BOS}{SH}system{EH}\n{sys_msg}\n{EOT}"
                    f"{SH}user{EH}\n{usr_msg}\n{EOT}"
                    f"{SH}assistant{EH}\n"
                )
            else:
                # Generic plain-text fallback
                prompt = f"System:\n{sys_msg}\n\nUser:\n{usr_msg}\n\nAssistant:\n"

        out = self.llm_client(
            prompt,
            do_sample=False,
            top_p=1.0,
            max_new_tokens=128,
            return_full_text=False,
            eos_token_id=getattr(self._tok, "eos_token_id", None),
        )
        return (out[0].get("generated_text") if isinstance(out, list) else str(out)).strip()


    # --- Parse the json from the llm ---
    def _parse_llm_json(self, text: str) -> dict:
        text = (text or "").strip()
        # Fast path
        if text.startswith("{") and text.endswith("}"):
            try:
                obj = json.loads(text)
                if isinstance(obj, dict) and "value" in obj and "confidence" in obj:
                    v = float(obj.get("value", 0.0)); c = float(obj.get("confidence", 0.0))
                    if math.isnan(v): v = 0.0
                    if math.isnan(c): c = 0.0
                    return {"value": v, "confidence": max(0.0, min(1.0, c))}
            except Exception:
                pass
        # Fallback: extract outermost {...}
        start = text.find("{"); end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end+1]
        else:
            return {"value": 0.0, "confidence": 0.0}

        try:
            obj = json.loads(text)
        except Exception:
            text2 = text.replace("\n", " ").replace("\r", " ").strip()
            try:
                obj = json.loads(text2)
            except Exception:
                return {"value": 0.0, "confidence": 0.0}

        if not isinstance(obj, dict):
            return {"value": 0.0, "confidence": 0.0}

        v = float(obj.get("value", 0.0))
        c = obj.get("confidence", 0.0)
        try:
            c = float(c)
        except Exception:
            c = 0.0
        if math.isnan(v): v = 0.0
        if math.isnan(c): c = 0.0
        c = max(0.0, min(1.0, c))
        return {"value": v, "confidence": c}


    # --- Observable bounds for summed paulis ---
    def _obs_bound(self, O: Any) -> float:
        """Return a conservative operator norm bound for O."""
        if isinstance(O, SparsePauliOp):
            coeffs = O.coeffs
        elif isinstance(O, str):
            coeffs = np.array([1.0], dtype=float)
        elif isinstance(O, list) and all(isinstance(s, str) for s in O):
            coeffs = np.ones(len(O), dtype=float)
        else:
            raise TypeError(f"Unsupported observable format: {type(O)}")
        return float(np.sum(np.abs(coeffs)))

    # --- Main LLM estimator ---
    def estimate_observable_llm(self, O: Any) -> Tuple[float, Optional[float]]:
        """Return (estimate, optional_confidence) via LLM."""
        if self.llm_client is None:
            raise RuntimeError("LLM client not set. Call set_llm(...) first.")

        bound = self._obs_bound(O)
        if not self.snapshots:
            return 0.0, 0.0

        # Build compact prompts for all snapshots
        prompts: list[str] = []
        for s in self.snapshots:
            enc = self._encode_snapshot(s)
            if self.llm_use_fast_prompt:
                prompts.append(self._build_llm_prompt_fast(str(O), enc))
            else:
                # fallback: flatten chat prompt to plain text
                msgs = self._build_llm_prompt(str(O), enc)
                sys_msg = next((m.get("content", "") for m in msgs if m.get("role") == "system"), "")
                usr_msg = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
                prompts.append(f"System:\n{sys_msg}\n\nUser:\n{usr_msg}\n\nAssistant:\n")

        vals: list[float] = []
        confs: list[float] = []

        # Run batched
        B = max(1, int(self.llm_batch_size))
        for i in range(0, len(prompts), B):
            chunk = prompts[i:i+B]
            out = self.llm_client(
                chunk,
                do_sample=False,
                top_p=1.0,
                max_new_tokens=self.llm_max_new_tokens,
                return_full_text=False,
                eos_token_id=getattr(self._tok, "eos_token_id", None),
            )
            # Pipeline returns list-of-lists when batched
            if isinstance(out, list) and out and isinstance(out[0], list):
                outs = [o[0].get("generated_text", "") for o in out]
            else:
                outs = [o.get("generated_text", "") for o in out]

            for txt in outs:
                parsed = self._parse_llm_json(txt)
                v = float(parsed.get("value", 0.0))
                c = float(parsed.get("confidence", 0.0))
                if math.isnan(v): v = 0.0
                v = max(-bound, min(bound, v))
                if math.isnan(c): c = 0.0
                c = max(0.0, min(1.0, c))
                vals.append(v); confs.append(c)

        arr_v = np.asarray(vals, dtype=float)
        arr_c = np.asarray(confs, dtype=float)
        mean_v = float(arr_v.mean()) if arr_v.size else 0.0
        mean_c = float(arr_c.mean()) if arr_c.size else 0.0
        return mean_v, mean_c


    # =======================================================================================



    # --- Unified single-observable API ---
    def predict(self, O: Any, method: Optional[Predictor] = None) -> Tuple[float, Optional[float]]:
        """Dispatch to classical or llm predictor."""
        method = method or self.default_predictor

        if method == "classical":
            return self.estimate_observable_classical(O)
        elif method == "llm":
            return self.estimate_observable_llm(O)
        else:
            raise ValueError(f"Unknown predictor: {method}")

    def predict_all(self, method: Optional[Predictor] = None,
                observables: Optional[List[Any]] = None) -> Dict[int, Tuple[float, Optional[float]]]:
        """
        Run predictions for all observables.
        - method: 'classical' | 'llm' | None (falls back to self.default_predictor)
        - observables: optional override list; defaults to self.observables
        Returns: {i: (estimate, optional_confidence)}
        """
        
        obs_list = observables if observables is not None else (self.observables or [])
        
        out: Dict[int, Tuple[float, Optional[float]]] = {}

        # Iterate over observable list and return the predictions
        for i, O in enumerate(obs_list):
            out[i] = self.predict(O, method=method)
            
        return out


    # ============================ Running Section ===============================

    def _serialize_snapshot(self, snap: dict) -> dict:
        out = dict(snap)
        for k in ("bases", "cliff_idx", "axes", "signs"):
            if k in out and hasattr(out[k], "tolist"):
                out[k] = out[k].tolist()
        return out
    
    def _save_or_load(self, path: str | None, op: str, *, merge: bool = True) -> dict | None:
        """
        Helper for save/load logic.
        op: "save" or "load"
        """
        from pathlib import Path
        import json
        p = Path(path) if path else Path(self.storage_path)
        if op == "save":
            p.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "meta": {
                    "n_qubits": self.n_qubits,
                    "scheme": self.scheme,
                    "default_predictor": self.default_predictor,
                },
                "snapshots": [self._serialize_snapshot(s) for s in self.snapshots],
                "observables": list(self.observables),
            }
            p.write_text(json.dumps(payload))
            return None
        elif op == "load":
            if not p.exists():
                return None
            data = json.loads(p.read_text())
            # Basic meta (optionally verify consistency)
            meta = data.get("meta", {})
            if "n_qubits" in meta and meta["n_qubits"] != self.n_qubits:
                raise ValueError("Loaded n_qubits does not match current instance.")
            if "scheme" in meta and meta["scheme"] != self.scheme:
                raise ValueError("Loaded scheme does not match current instance.")
            loaded_snaps = data.get("snapshots", [])
            loaded_obs   = data.get("observables", [])
            if merge:
                self.snapshots.extend(loaded_snaps)
                self.observables.extend(x for x in loaded_obs if x not in self.observables)
            else:
                self.snapshots = loaded_snaps
                self.observables = loaded_obs
            return data
        else:
            raise ValueError("Unknown op for _save_or_load")

    def save_now(self, path: str | None = None) -> None:
        self._save_or_load(path, "save")


    # --- Uncapped snapshot runner ---
    def run_snapshots(self) -> None:
        """Execute N snapshot circuits and append to self.snapshots."""
        for _ in range(self.num_snapshots):
            # Setup quantum circuit model - Sample bases
            bases = self.sample_bases()

            # ... - Build the circuit
            qc, meta = self.build_snapshot_circuit(bases)

            # ... - Transpile the circuit
            transpiled_qc = transpile(qc, self.backend)

            # Execute the model
            job = self.backend.run(transpiled_qc, shots=1)
            res = job.result().get_counts()
            bits = next(iter(res.keys()))
            
            # Store snapshot
            self.snapshots.append({**meta, "bits": bits})
        
    # --- Capped snapshot runner ---
    def run_snapshots_capped(
        self,
        N: int,
        *,
        max_shots_total: int | None = None,     # hard cap on total shots
        max_seconds_total: float | None = None, # wall-clock cap
        save_every: int = 500,                  # checkpoint cadence
        save_path: str | None = None            # override path
    ) -> int:
        """
        Collect up to N snapshots, but stop early if caps are reached.
        Returns the number of snapshots actually collected.
        """
        if self.backend is None:
            raise RuntimeError("Call set_backend first.")
        if self.state_prep is None:
            raise RuntimeError("Call set_state_prep first.")
        if N <= 0:
            return 0
    
        start = time.time()
        taken = 0
        shots_used = 0
    
        for _ in range(N):
            # cap: wall-clock
            if max_seconds_total is not None and (time.time() - start) >= max_seconds_total:
                break
            # cap: shots (we use 1 shot per snapshot)
            if max_shots_total is not None and shots_used >= max_shots_total:
                break
    
            bases = self.sample_bases()
            qc, meta = self.build_snapshot_circuit(bases)
            tqc = transpile(qc, self.backend)
            job = self.backend.run(tqc, shots=1)
            counts = job.result().get_counts()
            bits = next(iter(counts.keys()))
    
            self.snapshots.append({**meta, "bits": bits})
            taken += 1
            shots_used += 1
    
            if save_every and (taken % save_every == 0):
                self.save_now(save_path)
    
        # final checkpoint if we collected anything new
        if taken > 0:
            self.save_now(save_path)
    
        return taken


    # --- Resume runs from last snapshot ---
    def load_now(self, path: str | None = None, *, merge: bool = True) -> None:
        """Load snapshots + minimal meta from JSON; optionally merge with current memory."""
        self._save_or_load(path, "load", merge=merge)

    def resume_snapshots_capped(
        self,
        N: int,
        *,
        load_path: str | None = None,
        save_path: str | None = None,
        max_shots_total: int | None = None,
        max_seconds_total: float | None = None,
        save_every: int = 500
    ) -> int:
        """
        Load any existing snapshots, then append up to N more under the given caps.
        Saves periodically and at the end.
        """
        self.load_now(load_path, merge=True)
        taken = self.run_snapshots_capped(
            N,
            max_shots_total=max_shots_total,
            max_seconds_total=max_seconds_total,
            save_every=save_every,
            save_path=save_path
        )
        # Final save is already done in run_snapshots_capped; keeping this explicit is harmless
        self.save_now(save_path)
        return taken


    # ==================== Internal Helpers =====================
    def per_snapshot_values_classical(self, O: str) -> np.ndarray:
        """Per-snapshot contributions for O using the classical inverse."""
        if not isinstance(O, str) or len(O) != self.n_qubits or any(c not in "IXYZ" for c in O):
            raise ValueError("O must be a Pauli string over IXYZ with length n_qubits.")
        vals = []
        for snap in self.snapshots:
            sch = snap["scheme"]
            if sch in ("local-pauli", "local-clifford"):
                vals.append(self._inverse_one_snapshot(snap, O))
            elif sch == "global-clifford":
                vals.append(self._estimate_one_global_clifford(snap, O))
        return np.asarray(vals, dtype=float)

    def per_snapshot_values_llm(self, O: str) -> tuple[np.ndarray, np.ndarray]:
        """Per-snapshot values and confidences for O via LLM (batched)."""
        if self.llm_client is None:
            raise RuntimeError("LLM client not set. Call set_llm(...) first.")
        bound = self._obs_bound(O)
        if not self.snapshots:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)

        prompts: list[str] = []
        for s in self.snapshots:
            enc = self._encode_snapshot(s)
            if self.llm_use_fast_prompt:
                prompts.append(self._build_llm_prompt_fast(str(O), enc))
            else:
                msgs = self._build_llm_prompt(str(O), enc)
                sys_msg = next((m.get("content", "") for m in msgs if m.get("role") == "system"), "")
                usr_msg = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
                prompts.append(f"System:\n{sys_msg}\n\nUser:\n{usr_msg}\n\nAssistant:\n")

        vals: list[float] = []
        confs: list[float] = []
        B = max(1, int(self.llm_batch_size))
        for i in range(0, len(prompts), B):
            chunk = prompts[i:i+B]
            out = self.llm_client(
                chunk,
                do_sample=False,
                top_p=1.0,
                max_new_tokens=self.llm_max_new_tokens,
                return_full_text=False,
                eos_token_id=getattr(self._tok, "eos_token_id", None),
            )
            if isinstance(out, list) and out and isinstance(out[0], list):
                outs = [o[0].get("generated_text", "") for o in out]
            else:
                outs = [o.get("generated_text", "") for o in out]

            for txt in outs:
                parsed = self._parse_llm_json(txt)
                v = float(parsed.get("value", 0.0))
                c = float(parsed.get("confidence", 0.0))
                if math.isnan(v): v = 0.0
                v = max(-bound, min(bound, v))
                if math.isnan(c): c = 0.0
                c = max(0.0, min(1.0, c))
                vals.append(v); confs.append(c)

        return np.asarray(vals, dtype=float), np.asarray(confs, dtype=float)

    def _save_results(self, O: str, path: str, method: str,
                  vals: np.ndarray, confs: np.ndarray | None = None) -> None:
        meta = {
            "method": method, "observable": O,
            "n_qubits": self.n_qubits, "scheme": self.scheme, "N": int(vals.size),
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)   # <-- ensure folder exists
        if confs is not None:
            np.savez_compressed(p, vals=vals, confs=confs, meta=np.array([json.dumps(meta)], dtype=object))
        else:
            np.savez_compressed(p, vals=vals, meta=np.array([json.dumps(meta)], dtype=object))

    def save_results_classical(self, O: str, path: str) -> None:
        """Save per-snapshot classical contributions for observable O."""
        vals = self.per_snapshot_values_classical(O)
        self._save_results(O, path, "classical", vals)

    def save_results_llm(self, O: str, path: str) -> None:
        """Save per-snapshot LLM values & confidences for observable O."""
        vals, confs = self.per_snapshot_values_llm(O)
        self._save_results(O, path, "llm", vals, confs)

    def load_results(self, path: str) -> dict:
        """Load results file created by save_results_* and return dict with arrays+meta."""
        with np.load(path, allow_pickle=True) as f:
            meta = json.loads(f["meta"][0].item())
            out = {"meta": meta, "vals": f["vals"]}
            if "confs" in f.files:
                out["confs"] = f["confs"]
            return out


    # Plotting methods have been moved to ShadowTomoPlotter class below.
    

    def stats_summary(self, O: str) -> dict:
        """Return a small dict of useful stats for O (classical)."""
        vals = self.per_snapshot_values_classical(O)
        if vals.size == 0:
            return {"N": 0, "mean": 0.0, "se": 0.0, "zero_rate": 0.0}
        mean = float(vals.mean())
        se = float(vals.std(ddof=1)/math.sqrt(len(vals))) if len(vals) > 1 else 0.0
        zero_rate = float(np.mean(vals == 0.0))
        return {"N": int(len(vals)), "mean": mean, "se": se, "zero_rate": zero_rate}


    def stats_table(self, observables: list[str]) -> list[dict]:
        """Compute stats_summary for each O and return a list of dicts."""
        out = []
        for O in observables:
            row = {"O": O}
            row.update(self.stats_summary(O))
            out.append(row)
        return out


# ======================== Plotter Class ========================
class ShadowTomoPlotter:
    """
    Handles all plotting for a CTomo instance.
    """
    def __init__(self, tomo: CTomo):
        self.tomo = tomo

    # --- Saving helpers (individual SVGs + optional combined PDF) ---
    def _ensure_dir(self, save_dir: str | None) -> str:
        import os
        d = save_dir or "plots"
        os.makedirs(d, exist_ok=True)
        return d

    def _slug(self, text: str) -> str:
        return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(text)).strip("_")


    def plot_running_mean_from_file(self, path: str, *, save_dir: str | None = "plots") -> None:
        """Running mean ± 95% CI from a saved results file (classical or LLM)."""
        data = self.tomo.load_results(path)
        vals = data["vals"]
        if vals.size == 0:
            print("No data in file.")
            return
        import os
        k = np.arange(1, vals.size+1)
        mean = np.cumsum(vals) / k
        # simple rolling SE (unbiased) using incremental variance approx
        diffs = vals - mean
        sq = np.cumsum(diffs*diffs)
        se = np.zeros_like(mean)
        mask = k > 1
        se[mask] = np.sqrt(sq[mask] / (k[mask]-1)) / np.sqrt(k[mask])

        fig = plt.figure()
        plt.plot(k, mean, label="Running mean")
        upper = mean + 1.96*se
        lower = mean - 1.96*se
        plt.fill_between(k, lower, upper, alpha=0.2)
        m = data["meta"]
        plt.xlabel("Snapshots used")
        plt.ylabel(f"<{m['observable']}> estimate")
        plt.title(f"Running mean & 95% CI ({m['method']})")
        plt.legend()

        # Save individual vector output and optionally add to combined PDF
        d = self._ensure_dir(save_dir)
        fname = os.path.join(d, f"running_mean_{self._slug(m['observable'])}_{self._slug(m['method'])}.svg")
        fig.savefig(fname, format="svg", bbox_inches="tight")
        plt.show()

    def plot_hist_from_file(self, path: str, bins: int = 31, *, save_dir: str | None = "plots") -> None:
        """Histogram of per-snapshot contributions from a saved results file."""
        data = self.tomo.load_results(path)
        vals = data["vals"]
        if vals.size == 0:
            print("No data in file.")
            return
        import os
        fig = plt.figure()
        plt.hist(vals, bins=bins)
        m = data["meta"]
        plt.xlabel("Per-snapshot contribution")
        plt.ylabel("Count")
        plt.title(f"Snapshot contribution histogram for {m['observable']} ({m['method']})")

        d = self._ensure_dir(save_dir)
        fname = os.path.join(d, f"snapshot_hist_{self._slug(m['observable'])}_{self._slug(m['method'])}.svg")
        fig.savefig(fname, format="svg", bbox_inches="tight")
        plt.show()

    def plot_llm_confidence_from_file(self, path: str, bins: int = 20, *, save_dir: str | None = "plots") -> None:
        """Histogram of LLM confidences from a saved LLM results file."""
        data = self.tomo.load_results(path)
        if data["meta"]["method"] != "llm":
            print("Not an LLM results file.")
            return
        confs = data.get("confs")
        if confs is None or confs.size == 0:
            print("No confidences found.")
            return
        import os
        fig = plt.figure()
        plt.hist(confs, bins=bins, range=(0,1))
        m = data["meta"]
        plt.xlabel("Confidence")
        plt.ylabel("Count")
        plt.title(f"LLM confidence histogram for {m['observable']}")

        d = self._ensure_dir(save_dir)
        fname = os.path.join(d, f"llm_conf_hist_{self._slug(m['observable'])}.svg")
        fig.savefig(fname, format="svg", bbox_inches="tight")
        plt.show()

    def plot_snapshot_hist(self, O: str, bins: int = 31, *, save_dir: str | None = "plots") -> None:
        """Histogram of per-snapshot contributions for O (heavy tails check)."""
        vals = self.tomo.per_snapshot_values_classical(O)
        if vals.size == 0:
            print("No snapshots to plot.")
            return
        import os
        fig = plt.figure()
        plt.hist(vals, bins=bins)
        plt.xlabel("Per-snapshot contribution")
        plt.ylabel("Count")
        plt.title(f"Snapshot contribution histogram for {O}")

        d = self._ensure_dir(save_dir)
        fname = os.path.join(d, f"snapshot_hist_{self._slug(O)}.svg")
        fig.savefig(fname, format="svg", bbox_inches="tight")
        plt.show()

    def plot_basis_usage(self, *, save_dir: str | None = "plots") -> None:
        """Show how often Z/X/Y were effectively measured (local schemes)."""
        if not self.tomo.snapshots:
            print("No snapshots to plot.")
            return
        from collections import Counter
        import os
        counts = Counter()
        total = 0
        for s in self.tomo.snapshots:
            if s.get("axes") is None:
                continue
            for a in s["axes"]:
                counts[a] += 1
                total += 1
        if total == 0:
            print("No local-scheme axis data available.")
            return
        labels = ["Z","X","Y"]
        heights = [counts.get(0,0), counts.get(1,0), counts.get(2,0)]
        fig = plt.figure()
        plt.bar(labels, heights)
        plt.xlabel("Effective measured axis")
        plt.ylabel("Count")
        plt.title("Axis usage across local snapshots")

        d = self._ensure_dir(save_dir)
        fname = os.path.join(d, "basis_usage.svg")
        fig.savefig(fname, format="svg", bbox_inches="tight")
        plt.show()

    def plot_bit_marginals(self, *, save_dir: str | None = "plots") -> None:
        """Per-qubit probability of measuring 1 across all snapshots."""
        if not self.tomo.snapshots:
            print("No snapshots to plot.")
            return
        import os
        ones = np.zeros(self.tomo.n_qubits, dtype=float)
        N = 0
        for s in self.tomo.snapshots:
            bits = s["bits"]
            # rightmost bit is qubit 0
            for q in range(self.tomo.n_qubits):
                ones[q] += 1.0 if bits[-1 - q] == "1" else 0.0
            N += 1
        if N == 0:
            print("No snapshots to plot.")
            return
        p1 = ones / N
        fig = plt.figure()
        plt.bar(np.arange(self.tomo.n_qubits), p1)
        plt.xlabel("Qubit index")
        plt.ylabel("P(bit=1)")
        plt.title("Per-qubit bit marginals")

        d = self._ensure_dir(save_dir)
        fname = os.path.join(d, "bit_marginals.svg")
        fig.savefig(fname, format="svg", bbox_inches="tight")
        plt.show()

    def plot_llm_vs_classical(self, observables: list[str], *, save_dir: str | None = "plots") -> None:
        """Scatter of LLM mean vs classical mean for each observable in list."""
        xs, ys = [], []
        labels = []
        for O in observables:
            # classical
            c_mean, _ = self.tomo.estimate_observable_classical(O)
            # llm (guard if no client)
            if self.tomo.llm_client is None:
                print("LLM client not set; skipping.")
                return
            l_mean, _ = self.tomo.estimate_observable_llm(O)
            xs.append(c_mean); ys.append(l_mean); labels.append(O)
        if not xs:
            print("No data to plot.")
            return
        mn = min(xs+ys); mx = max(xs+ys)
        pad = 0.05*(mx - mn + 1e-9)
        import os
        fig = plt.figure()
        plt.scatter(xs, ys)
        plt.plot([mn-pad, mx+pad], [mn-pad, mx+pad])
        plt.xlabel("Classical estimate")
        plt.ylabel("LLM estimate")
        plt.title("LLM vs Classical")

        d = self._ensure_dir(save_dir)
        fname = os.path.join(d, "llm_vs_classical.svg")
        fig.savefig(fname, format="svg", bbox_inches="tight")
        plt.show()

    def plot_stats_dashboard(self, observables: list[str], *, save_dir: str | None = "plots") -> None:
        """
        Plot a dashboard of summary statistics for a list of observables.
        Shows mean, standard error, and zero rate for each observable.
        """
        # Compute stats for each observable
        stats = self.tomo.stats_table(observables)
        if not stats:
            print("No statistics to plot.")
            return
        import os
        labels = [row["O"] for row in stats]
        means = [row["mean"] for row in stats]
        ses = [row["se"] for row in stats]
        zero_rates = [row["zero_rate"] for row in stats]

        x = np.arange(len(labels))
        fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

        # Mean
        axs[0].bar(x, means, yerr=ses, capsize=4)
        axs[0].set_ylabel("Mean ± SE")
        axs[0].set_title("Mean and Standard Error per Observable")

        # Standard error
        axs[1].bar(x, ses)
        axs[1].set_ylabel("Standard Error")
        axs[1].set_title("Standard Error per Observable")

        # Zero rate
        axs[2].bar(x, zero_rates)
        axs[2].set_ylabel("Zero Rate")
        axs[2].set_xlabel("Observable")
        axs[2].set_title("Fraction of Zero Contributions per Observable")
        axs[2].set_xticks(x)
        axs[2].set_xticklabels(labels, rotation=45, ha="right")

        plt.tight_layout()

        d = self._ensure_dir(save_dir)
        fname = os.path.join(d, "stats_dashboard.svg")
        fig.savefig(fname, format="svg", bbox_inches="tight")
        plt.show()

    def plot_all_vector(
        self,
        *,
        save_dir: str | None = "plots",
        classical_path: str | None = None,
        llm_path: str | None = None,
        O_for_hist: str | None = None,
        compare_observables: list[str] | None = None,
        stats_observables: list[str] | None = None,
        filename: str = "all_plots.svg",
        bins_hist: int = 31,
        bins_conf: int = 20
    ) -> None:
        """
        Create a single vector figure that combines multiple panels:
          - Running mean (from classical_path)
          - Snapshot histogram (from classical_path)
          - LLM confidence histogram (if llm_path is provided and method=='llm')
          - Bit marginals (from in-memory snapshots)
          - Basis usage (from in-memory snapshots)
          - LLM vs Classical scatter (if compare_observables provided)
          - Stats dashboard (if stats_observables provided) rendered as a nested 3×1 subgrid

        All panels included only if the corresponding inputs are provided/available.
        Saves one SVG file at `save_dir/filename`.
        """
        import os
        from matplotlib import pyplot as plt
        import numpy as np

        panels = []

        # Prepare data for panels
        classical_data = self.tomo.load_results(classical_path) if classical_path else None
        if classical_data is not None:
            panels.append("running_mean")
            panels.append("snap_hist")

        llm_data = self.tomo.load_results(llm_path) if llm_path else None
        if llm_data is not None and llm_data.get("meta", {}).get("method") == "llm":
            panels.append("llm_conf")

        # Always possible from memory (if snapshots exist)
        if getattr(self.tomo, "snapshots", None):
            panels.append("bit_marginals")
            panels.append("basis_usage")

        if compare_observables:
            panels.append("llm_vs_classical")

        if stats_observables:
            panels.append("stats_dashboard")

        if not panels:
            print("Nothing to plot for combined figure.")
            return

        # Create subplots (1 column)
        fig, axs = plt.subplots(len(panels), 1, figsize=(10, 3.2 * len(panels)), constrained_layout=True)
        if len(panels) == 1:
            axs = [axs]  # normalize to list

        ax_idx = 0

        # ---- Running mean panel ----
        if "running_mean" in panels:
            vals = classical_data["vals"]
            k = np.arange(1, vals.size+1)
            mean = np.cumsum(vals) / k
            diffs = vals - mean
            sq = np.cumsum(diffs*diffs)
            se = np.zeros_like(mean)
            mask = k > 1
            se[mask] = np.sqrt(sq[mask] / (k[mask]-1)) / np.sqrt(k[mask])
            ax = axs[ax_idx]; ax_idx += 1
            ax.plot(k, mean, label="Running mean")
            upper = mean + 1.96*se
            lower = mean - 1.96*se
            ax.fill_between(k, lower, upper, alpha=0.2)
            m = classical_data["meta"]
            ax.set_xlabel("Snapshots used")
            ax.set_ylabel(f"<{m['observable']}> estimate")
            ax.set_title(f"Running mean & 95% CI ({m['method']})")
            ax.legend()

        # ---- Snapshot histogram panel ----
        if "snap_hist" in panels:
            vals = classical_data["vals"]
            ax = axs[ax_idx]; ax_idx += 1
            ax.hist(vals, bins=bins_hist)
            m = classical_data["meta"]
            ax.set_xlabel("Per-snapshot contribution")
            ax.set_ylabel("Count")
            ax.set_title(f"Snapshot contribution histogram for {m['observable']} ({m['method']})")

        # ---- LLM confidence histogram panel ----
        if "llm_conf" in panels:
            confs = llm_data.get("confs")
            ax = axs[ax_idx]; ax_idx += 1
            if confs is None or confs.size == 0:
                ax.text(0.5, 0.5, "No confidences found", ha="center", va="center")
            else:
                ax.hist(confs, bins=bins_conf, range=(0,1))
                ax.set_xlabel("Confidence")
                ax.set_ylabel("Count")
                ax.set_title(f"LLM confidence histogram for {llm_data['meta']['observable']}")

        # ---- Bit marginals panel ----
        if "bit_marginals" in panels:
            ones = np.zeros(self.tomo.n_qubits, dtype=float)
            N = 0
            for s in self.tomo.snapshots:
                bits = s["bits"]
                for q in range(self.tomo.n_qubits):
                    ones[q] += 1.0 if bits[-1 - q] == "1" else 0.0
                N += 1
            ax = axs[ax_idx]; ax_idx += 1
            if N == 0:
                ax.text(0.5, 0.5, "No snapshots", ha="center", va="center")
            else:
                p1 = ones / N
                ax.bar(np.arange(self.tomo.n_qubits), p1)
                ax.set_xlabel("Qubit index")
                ax.set_ylabel("P(bit=1)")
                ax.set_title("Per-qubit bit marginals")

        # ---- Basis usage panel ----
        if "basis_usage" in panels:
            from collections import Counter
            counts = Counter()
            total = 0
            for s in self.tomo.snapshots:
                if s.get("axes") is None:
                    continue
                for a in s["axes"]:
                    counts[a] += 1
                    total += 1
            ax = axs[ax_idx]; ax_idx += 1
            if total == 0:
                ax.text(0.5, 0.5, "No local-scheme axis data", ha="center", va="center")
            else:
                labels = ["Z","X","Y"]
                heights = [counts.get(0,0), counts.get(1,0), counts.get(2,0)]
                ax.bar(labels, heights)
                ax.set_xlabel("Effective measured axis")
                ax.set_ylabel("Count")
                ax.set_title("Axis usage across local snapshots")

        # ---- LLM vs Classical scatter panel ----
        if "llm_vs_classical" in panels:
            xs, ys = [], []
            for O in (compare_observables or []):
                c_mean, _ = self.tomo.estimate_observable_classical(O)
                if self.tomo.llm_client is None:
                    continue
                l_mean, _ = self.tomo.estimate_observable_llm(O)
                xs.append(c_mean); ys.append(l_mean)
            ax = axs[ax_idx]; ax_idx += 1
            if not xs:
                ax.text(0.5, 0.5, "No data for LLM vs Classical", ha="center", va="center")
            else:
                mn = min(xs+ys); mx = max(xs+ys)
                pad = 0.05*(mx - mn + 1e-9)
                ax.scatter(xs, ys)
                ax.plot([mn-pad, mx+pad], [mn-pad, mx+pad])
                ax.set_xlabel("Classical estimate")
                ax.set_ylabel("LLM estimate")
                ax.set_title("LLM vs Classical")

        # ---- Stats dashboard panel (nested subgrid) ----
        if "stats_dashboard" in panels:
            stats = self.tomo.stats_table(stats_observables or [])
            ax_outer = axs[ax_idx]; ax_idx += 1
            if not stats:
                ax_outer.text(0.5, 0.5, "No statistics to plot.", ha="center", va="center")
            else:
                labels = [row["O"] for row in stats]
                means = [row["mean"] for row in stats]
                ses = [row["se"] for row in stats]
                zero_rates = [row["zero_rate"] for row in stats]

                gs = ax_outer.get_subplotspec().subgridspec(3, 1, hspace=0.35)
                ax0 = fig.add_subplot(gs[0])
                ax1 = fig.add_subplot(gs[1])
                ax2 = fig.add_subplot(gs[2])

                x = np.arange(len(labels))
                ax0.bar(x, means, yerr=ses, capsize=4)
                ax0.set_ylabel("Mean ± SE")
                ax0.set_title("Mean and Standard Error per Observable")

                ax1.bar(x, ses)
                ax1.set_ylabel("Standard Error")
                ax1.set_title("Standard Error per Observable")

                ax2.bar(x, zero_rates)
                ax2.set_ylabel("Zero Rate")
                ax2.set_xlabel("Observable")
                ax2.set_title("Fraction of Zero Contributions per Observable")
                ax2.set_xticks(x)
                ax2.set_xticklabels(labels, rotation=45, ha="right")

                # Hide the outer placeholder axis
                ax_outer.set_visible(False)

        # Save the combined SVG
        d = self._ensure_dir(save_dir)
        out_path = os.path.join(d, filename)
        fig.savefig(out_path, format="svg", bbox_inches="tight")
        plt.show()