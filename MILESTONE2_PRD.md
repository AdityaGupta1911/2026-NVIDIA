# 🚀 Product Requirements Document (PRD)
## NVIDIA iQuHACK 2026 Challenge – Milestone 2
**Solo:** Aditya Kumar Gupta | **Phase 1 Submission**

---

## 1. Technical Role Assignments (Ensemble Orchestration)
In alignment with the **"Vibe Coding" track**, our project utilizes a multi-agent orchestration strategy. The Project Lead manages specialized AI personas to fulfill the following PIC (Person In Charge) roles:

| Role | Lead Persona | Primary Responsibilities |
| :--- | :--- | :--- |
| **Project Lead** | **Aditya Kumar Gupta** | Strategy, orchestration, milestone approvals, and judge communication. |
| **GPU Acceleration PIC** | **DeepSeek-V3** | CUDA-Q kernel optimization, TensorNet scaling, and Brev resource management. |
| **Quality Assurance PIC** | **Claude-3.5** | Physics validation ($H$ vs $E$ mapping), unit testing, and mathematical auditing. |
| **Technical Marketing PIC** | **Kimi-Search** | Research synthesis, README documentation, and performance visualization. |

---

## 2. Research Foundation & Algorithmic Choice
* **Selected Approach:** Digitized Counterdiabatic Quantum Optimization (DCQO) enhanced with Memetic Tabu Search (MTS).
* **Core Reference:** *Karamlou et al., "Scaling advantage with quantum-enhanced memetic tabu search for LABS," arXiv:2511.04553 (2025).*

**Rationale:** We utilize the **Counterdiabatic protocol** to generate high-quality quantum seeds. This approach is chosen because it significantly reduces circuit depth compared to traditional QAOA, making it ideal for the $N=40$ scale-up on the **NVIDIA A10 GPU**. These seeds provide a biased starting population that accelerates the convergence of our classical Tabu Search.



---

## 3. Verification & QA Strategy
To ensure mathematical integrity before scaling to $N=40$, our QA persona (**Claude**) implemented a rigorous validation suite for $N=20$:

* **Hamiltonian Mapping Audit:** Verified the mathematical consistency of $H = -2E - N(N-1)$.
* **Interaction Term Validation:** Confirmed the generation of exactly 90 (2-body) and 1050 (4-body) terms for $N=20$, matching the expansion of the LABS objective function.
* **Sign Convention Check:** Corrected a sign-inversion error to ensure the Tabu Search successfully minimizes the Hamiltonian to maximize the sequence Merit Factor.

---

## 4. GPU Acceleration & Execution Plan
**Hardware Target:** NVIDIA A10 (24GB VRAM) via Brev.nvidia.com

### **Scaling Strategy**
* **State Vector ($N \le 30$):** High-fidelity simulation using the `nvidia` CUDA-Q backend.
* **Tensor Networks ($N > 30$):** Switching to the `tensornet` (Matrix Product State) backend to bypass the exponential memory wall of $N=40$.

### **Optimization Tactics**
* **Vectorized Tabu Search:** Utilizing NumPy/CuPy sparse matrices to accelerate the classical refinement loop.
* **Bond Dimension Management:** Setting `max_bond_dimension=2000` to ensure stability within the 24GB VRAM limit.



---

## 5. Resource Management & Budget Allocation
**Total Allocation:** $20.00 USD (Brev Credits)

| Phase | Budget | Purpose | Instance Type |
| :--- | :--- | :--- | :--- |
| **Dev & Tuning** | $5.00 | $N=30$ validation and backend parameter tuning. | A10 (Spot/On-Demand) |
| **Production** | $10.00 | High-shot $N=40$ final benchmark runs. | A10 (On-Demand) |
| **Buffer** | $5.00 | Contingency for debugging and overhead. | Variable |

> [!IMPORTANT]
> **Anti-Zombie Protocol:** Implementation of 30-minute automated instance health checks and mandatory shutdown of the A10 instance immediately following the $N=40$ data collection.

---

## 6. Success Metrics
* **Logic Accuracy:** 100% pass rate on `validate_implementation` unit tests (Passed at $N=20$).
* **Merit Factor (MF):** Target $MF > 1.0$ for $N=40$, demonstrating a clear improvement over random initialization.
* **Scaling:** Successful execution of the $N=40$ circuit within a 300-second time-to-solution window using TensorNet.

---

## 7. Risk Mitigation
* **Risk:** Memory overflow at $N=40$.
    * *Mitigation:* Strategic use of `cudaq.set_target("tensornet")` with a bond dimension cutoff.
* **Risk:** Discrepancy in energy results.
    * *Mitigation:* Re-running the baseline $N=20$ comparison to ensure logic consistency during backend transitions.

---

**Signed:** *Project Lead (User)* **Technical Review:** *QA PIC, GPU PIC, Marketing PIC*
