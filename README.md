<br />

<h1 align="center">Motivation Driven Reinforcement
Learning With MiRo</h3>

  <p align="center">
    Motivation based algorithms are indispensable for robots to train animal-like behaviours. As reinforcement learning becomes more advanced over the years, it's possible for robots to navigate through a motivation and balance multiple motivations. This project focuses on balancing MiRo's two needs, hunger and thirst, and train MiRo to reach food and water in a maze by itself. Here I built an editable maze with multiple motivations inside. MiRo can successfully navigate through the needs and balance its motivation in a real-world environment. The Phase-Based algorithm works to help MiRo switch between it's needs smoothly. In general, this method can be applied to all types of take which need to fulfil multiple independent motivations simultaneously.



<!-- GETTING STARTED -->
## Getting Started


### Prerequisites

tqdm, matplotlib, and tkinter are used in the project.
* tqdm
  ```sh
  pip install tqdm
  ```
* matplotlib
  ```sh
  pip install matplotlib
  ```
    ```
* tkinter
  ```sh
  pip install tkinter
  ```
* To run the Model with MiRo, WSL-ROS needs to be installed, or connect to a remote desktop from: https://www.sheffield.ac.uk/findapc/rdp/room/52/pcs
* Otherwise, the virtual model can be run without the installation of ROS.

### Run the Model with MiRo
1. Clone the repo
   ```sh
   git clone https://github.com/SherimaX/Dissertation.git
   ```
2. Change the current working directory to the `Dissertation` folder
   ```sh
   cd Dissertation
   ```
3. generate a new world from `t-maze.pnm`
   ```sh
   python3 world_generator.py
   ```
4. Copy the generated world to ROS
   ```sh
   cp t-maze.world ~/mdk/sim/worlds
   ```
5. Copy the wall, water, food blocks, and the ground plane to ROS
   ```sh
   cp -r {water/,food/,my_ground_plane/,cube/} ~/mdk/sim/worlds
   ```
6. Start the ROS server in a new terminal tab
   ```sh
   roscore
   ```
6. Start up the gazebo world in a new terminal tab
   ```sh
   sh launch_sim_rosbridge.sh t-maze
   ```
  
7. Run agent.py 
   ```sh
   python3 agent.py
   ```
   
### Run the Model without MiRo
1. Clone the repo
   ```sh
   git clone https://github.com/SherimaX/Dissertation.git
   ```
2. Run simulation.py 
   ```sh
   python3 simulation.py
   ``` 

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<br>
