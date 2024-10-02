import glob
import io
import base64
import imageio
import gymnasium as gym
import torch
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder
from agent import Agent  # Make sure this is the correct import for your Agent class

def load_checkpoint(agent, filepath):
    agent.local_qnetwork.load_state_dict(torch.load(filepath))
    agent.local_qnetwork.eval()  # Set the model to evaluation mode

def show_video_of_model(agent, env_name):
    env = gym.make(env_name, render_mode='rgb_array')
    state, _ = env.reset()
    done = False
    frames = []
    while not done:
        frame = env.render()
        frames.append(frame)
        action = agent.act(state)
        state, reward, done, _, _ = env.step(action)
    env.close()
    imageio.mimsave('video.mp4', frames, fps=30)

# Initialize your agent
state_size = 8  # Adjust this based on your environment
action_size = 4  # Adjust this based on your environment
agent = Agent(state_size, action_size)

# Load the checkpoint
checkpoint_path = 'checkpoint.pth'  # Update this if your path is different
load_checkpoint(agent, checkpoint_path)

# Run the video generation
show_video_of_model(agent, 'LunarLander-v2')

def show_video():
    mp4list = glob.glob('*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        display(HTML(data='''<video alt="test" autoplay
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
    else:
        print("Could not find video")

show_video()





