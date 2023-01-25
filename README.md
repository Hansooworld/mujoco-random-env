# mujoco-random-env
Required Version

```python
gym == 0.23.1
mujoco_py == 2.1.2.14
mujoco == 2.2.2
```
</br>
You can randomly change model's parameter (e.g. agent's mass, friction, ...) without any library. 

You just need to insert randomness properties when you called the envrionment.

To randomly change the agent's mass,
```python
  env = AntRandomEnvClass(rand_mass=[1,5], rand_fric=None, render_mode=None, VERBOSE=True)
```
To randomly change the friction together,
```python
  env = AntRandomEnvClass(rand_mass=[1,5], rand_fric=[0.2, 1.0], render_mode=None, VERBOSE=True)
```
</br>
I also added the environment with the box to make available to apply meta reinforcement learning conveniently.

```python
  env = AntRandomEnvClassWithBox(rand_mass=[1,5], rand_fric=None, render_mode=None, VERBOSE=True)
```
The agent with the box looks like the following image.

![image](https://user-images.githubusercontent.com/77337434/193408761-a772c2a1-71f1-4ab5-b2ca-0e377bc21a76.png)
