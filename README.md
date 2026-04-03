# Cooperative Multi-UAV Formation Control via Decoupled MADDPG

### 1. Introduction & Objectives
The deployment of Unmanned Aerial Vehicle (UAV) swarms has shifted from executing simple, isolated tasks to performing highly coordinated, cooperative missions. To operate effectively in dynamic real-world scenarios, UAV swarms must be capable of maintaining strict geometric formations while navigating continuous space. 

This research builds upon a custom-designed, fully simulated multi-agent UAV environment. In our foundational work, the common goal of the multi-agent system was to cooperatively pass over a set of distributed waypoints with maximum frequency using Proximal Policy Optimization (PPO). 

In this current project, we dramatically evolve the environment's complexity by shifting the swarm's objective from decentralized waypoint navigation to rigid, cooperative formation control. The system uses the Multi-Agent Deep Deterministic Policy Gradient (MADDPG) framework to achieve highly precise, continuous formation keeping (Line, Column, V-Shape, and Circle).

---

### 2. Algorithmic Architecture: Decoupled MADDPG
To solve the complex continuous control required for formation keeping, we implemented Centralized Training with Decentralized Execution (CTDE) using MADDPG. During development, we encountered severe policy collapse when multiplexing all agents through a single Actor network, as the dominant gradients of the followers caused the leader to swerve erratically. 

To achieve global optima and absolute stability, we developed a **Decoupled "Hive Mind" Architecture**:
* **The Static Anchor (Leader):** Agent 0 is strictly designated as the formation anchor. It utilizes a dedicated **Leader Actor** network. To ensure cooperative behavior, the Leader's loss function is mathematically bound to the average of all centralized Critics, forcing it to maintain cruise velocity for the benefit of the entire swarm.
* **The Follower Swarm:** Agents 1 through N share a single **Follower Actor** network. This shared parameter space allows the followers to instantly share knowledge on how to dynamically brake, accelerate, and steer into their assigned geometric slots.
* **Centralized Critics:** During training, each agent utilizes a dedicated Critic network that observes the concatenated global state and actions of the entire swarm, allowing them to evaluate individual slot performance.

---

### 3. Environment Physics & Reward Shaping
The environment handles complex flight kinematics and continuous space mapping. To stabilize the learning gradients, we engineered highly specific reward mechanisms:

* **Torus-Safe Geometry:** The environment simulates an infinite continuous space by wrapping boundaries (Torus topology). All Bipartite Matching algorithms and collision detection matrices strictly use modulo mathematics to prevent catastrophic gradient spikes when drones cross the map edges.
* **The "Gravity Well" Slot Reward:** A two-part gradient pulls followers to their slots. A weak linear gradient (20%) provides global direction across the map, while an aggressive exponential curve (80%) based on the formation tolerance forces the drones to lock perfectly into their precise $(X,Y)$ coordinate.
* **The "Safe-Bubble" Repulsion:** To prevent the "Zig-Zag Trap" (where drones break formation to escape continuous repulsion penalties from neighbors), the collision penalty only triggers if a drone breaches 75% of the intended safe formation distance.
* **The "Alpha" Velocity Reward:** To solve the "Lazy Anchor Paradox" (where the leader stands still to artificially maximize the swarm's positional score), the Leader is immune to collision penalties and receives a massive positive reward solely for matching the target cruise speed, forcing it to drag the swarm forward.

---

### 4. State and Action Spaces
To fully support the Centralized Training with Decentralized Execution (CTDE) paradigm, the environment strictly defines what physical limits the agents can control and what localized information they can perceive.

**The Continuous Action Space:**
Each UAV utilizes a continuous 2D action space representing its flight control limits. Instead of absolute movement, the networks output normalized bounded vectors $[-1, 1]$ which are scaled to the drone's specific physical kinematics:
* **Acceleration (Throttle):** Scaled and clamped between $[0, \text{max\_accel}]$. Drones cannot fly in reverse; they must physically turn to change direction.
* **Angular Velocity (Steering):** Scaled and clamped between $[-\text{max\_omega}, \text{max\_omega}]$, defining the maximum turning radius in radians per second.

**The Local Observation Space (Decentralized State):**
To ensure the policy can be executed autonomously, an individual agent $i$ cannot see the global coordinates of the map. Instead, its observation vector $o_i$ is composed entirely of egocentric and relative measurements:
1.  **Self-Awareness:** The agent's own normalized velocity, its current heading vector $[\cos(\theta), \sin(\theta)]$, and the global target directional vector.
2.  **Anchor Awareness:** The normalized, Torus-wrapped relative distance vector $(X, Y)$ to the Leader UAV.
3.  **Swarm Awareness:** A flattened array containing the relative positions and relative velocities of all other $N-1$ agents in the swarm.

---

### 5. Dynamic Geometric Formation Logic (Slot Assignment)
To make the swarm robust to disturbances and dynamic spawning, followers are *not* hardcoded to specific positions in the formation. Instead, the environment utilizes a dynamic "Slot" generation and matching algorithm.

**Geometric Slot Matrix Generation:**
Every simulation step, the environment calculates a matrix of ideal geometric coordinates (`valid_slots`) relative to the Leader's current position and heading.
* **COLUMN:** Slots are generated at scalar multiples of the formation distance directly along the negative target vector.
* **LINE:** A lateral vector perpendicular to the heading is calculated, and slots are spawned in alternating left/right sequence based on odd/even indices.
* **V-SHAPE:** The target heading angle is swept back by $135^\circ$ ($3\pi/4$ rad) to create two directional arm vectors. Slots are populated in alternating left/right steps along these swept arms.
* **CIRCLE:** The space around the leader is divided into equal angular slices, rotated smoothly by the target heading to maintain orientation.

---

### 6. Mathematical Formulation of Geometric Slot Matrices
The environment dynamically computes target coordinate slots for the followers relative to the leader's position and orientation. Let $N$ be the total number of UAVs, making the number of followers $N_f = N - 1$. 

For all calculations, we define the following global formation parameters:
* $d$: The target formation keeping distance.
* $\vec{T} = [T_x, T_y]$: The normalized target heading vector of the leader.
* $\theta_T$: The leader's heading angle in radians, derived as $\theta_T = \text{atan2}(T_y, T_x)$.
* $i$: The index of the unassigned follower slot, where $i \in \{0, 1, \dots, N_f - 1\}$.

The slot positions, $\vec{S}_i$, are calculated as relative coordinate vectors from the leader. The absolute target coordinate on the map for a slot is simply the leader's current Torus-wrapped $(X, Y)$ position plus $\vec{S}_i$.

**1. Column Formation (Trail)**
* **Row Multiplier:** $r_i = i + 1$
* **Directional Vector:** The inverse of the target heading, $-\vec{T}$.
* **Slot Equation:** $$\vec{S}_i = r_i \cdot d \cdot (-\vec{T})$$

**2. Line Formation (Abreast)**
* **Lateral Vector ($\vec{L}$):** A vector perpendicular to the heading $\vec{T}$. Calculated via a $90^\circ$ rotation matrix: $\vec{L} = [-T_y, T_x]$.
* **Row Multiplier:** $r_i = \lfloor i / 2 \rfloor + 1$
* **Alternating Sign ($s_i$):** $+1$ for even indices (right side), $-1$ for odd indices (left side).
* **Slot Equation:** $$\vec{S}_i = r_i \cdot d \cdot s_i \cdot \vec{L}$$

**3. V-Shape Formation**
* **Sweep Angle ($\phi$):** The arms are swept back by $135^\circ$ ($3\pi/4$ radians) relative to the forward heading $\theta_T$.
* **Arm Directional Vectors:**
    * Arm 1 (Even slots): $\vec{A}_1 = [\cos(\theta_T + \phi), \sin(\theta_T + \phi)]$
    * Arm 2 (Odd slots): $\vec{A}_2 = [\cos(\theta_T - \phi), \sin(\theta_T - \phi)]$
* **Row Multiplier:** $r_i = \lfloor i / 2 \rfloor + 1$
* **Slot Equation:**
    $$\vec{S}_i = \begin{cases} r_i \cdot d \cdot \vec{A}_1 & \text{if } i \text{ is even} \\ r_i \cdot d \cdot \vec{A}_2 & \text{if } i \text{ is odd} \end{cases}$$

**4. Circle Formation**
* **Angular Slices ($\alpha_i$):** The circle is divided into $N_f$ equal segments: $\alpha_i = i \cdot \left(\frac{2\pi}{N_f}\right)$.
* **Offset Angle ($\theta_i$):** The slice angle is offset by the leader's global heading: $\theta_i = \alpha_i + \theta_T$.
* **Slot Equation:** $$\vec{S}_i = d \cdot [\cos(\theta_i), \sin(\theta_i)]$$

---

### 7. Torus-Safe Bipartite Matching
To ensure a flawless 1-to-1 mapping without mid-air collisions, the environment employs a Greedy Bipartite Assignment algorithm calculated over a Torus topology.

**Part 1: The Torus Distance Matrix (The Modulo Trick)**
Standard Euclidean distance math fails on a Torus map. To fix this, the code computes the difference vector using a specific Modulo Arithmetic formula for both the $X$ and $Y$ axes independently:

$$\Delta_{torus} = \left( \left( \text{Pos}_{follower} - \text{Pos}_{slot} + \frac{\text{Grid}}{2} \right) \pmod{\text{Grid}} \right) - \frac{\text{Grid}}{2}$$

By applying this math to the $X$ and $Y$ coordinates of every drone and every slot simultaneously, the code generates a matrix of the true shortest-path distances across the Torus.

**Part 2: The Greedy Bipartite Assignment**
1. **Global Minimum Search:** The algorithm scans the entire matrix and finds the single smallest distance.
2. **Assignment:** It locks that Follower to that Slot.
3. **Matrix Elimination (The Infinity Cross-Out):** To prevent double assignments, the algorithm overwrites the assigned Follower's entire row and the assigned Slot's entire column with `Infinity` ($\infty$).
4. **Iteration:** It scans the matrix for the *next* smallest number and repeats the process until all followers are locked in.

---

### 8. Codebase Overview
The project is modularized into several core operational scripts:

* **`train_MADDPG.py`**: The centralized training script. It dynamically builds the Decoupled Actor and Critic networks, manages the GPU-native ReplayBuffer, scales exploration noise to exact physical limits, computes batched backpropagation, and generates unique timestamped logging files.
* **`uav_environment.py`**: The physics and geometry engine. It calculates target vectors, updates Torus-safe continuous flight kinematics, applies action clamping, calculates bipartite-matched target slots, and computes the engineered reward gradients.
* **`evaluate_MADDPG.py`**: The decentralized execution script. It loads the decoupled trained weights (`trained_leader.pth` and `trained_follower.pth`), routes the drones to their respective brains, disables exploration noise, and runs a purely deterministic, real-time visual simulation of the learned policy.
* **`test_formations.py`**: A dedicated unit testing suite utilizing `unittest.mock.patch`. It mathematically intercepts the Bipartite Matching outputs and generates Matplotlib graphs to explicitly validate the geometrical spawn points of the Line, Column, V-Shape, and Circle slots without requiring the physics engine to run.
* **`config_*.json`**: Configuration files dictating swarm parameters, physics bounds, and training hyperparameters (2000 epochs, 300 steps, noise decay of 0.997).

```text
Project_Root/
│
├── configs/
│   ├── config_CIRCLE.json
│   ├── config_COLUMN.json
│   ├── config_LINE.json
│   └── config_V_SHAPE.json
│
├── modules/
│   ├── actor.py               
│   ├── critic.py              
│   ├── direction.py           
│   ├── fomation_type.py       
│   └── utils.py               
│
├── results/                   
│   └── [Formation_Name]/      
│
├── evaluate_MADDPG.py         
├── test_formations.py         
├── train_MADDPG.py            
├── uav_environment.py         
└── README.md
```

---

### 9. Executing the System
To validate geometries: 
Run python test_formations.py to generate visual plots of the slot matrices in the results/formation_test_plots directory. This allows you to mathematically verify the shape configurations before spending time running the physics simulation.

To train a policy: 
Run python train_MADDPG.py. The script will automatically detect your hardware (CUDA/CPU), train the decoupled networks over the specified epochs, and export the .pth weight files and learning curves to the results/ directory.

To evaluate the swarm: 
Run python evaluate_MADDPG.py. The script will initialize the deterministic Actor networks and open a live Matplotlib render window. It automatically captures screenshots at the start, middle, and end of the evaluation and stitches them into a final [Formation]_progress_showcase.png output.
