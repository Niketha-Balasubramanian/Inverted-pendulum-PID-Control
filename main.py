import gymnasium as gym
import numpy as np
import time

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0
        self.counter = 0 # Added for PWM logic

    def get_action(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        self.prev_error = error
        
        # --- PWM LOGIC ---
        # Instead of 100% force, we scale the 'probability' of a push.
        # This simulates a 'gentle' motor.
        self.counter = (self.counter + 1) % 5
        threshold = min(abs(output) * 10, 5) # Scale output to a 0-5 range
        
        if self.counter < threshold:
            return 1 if output > 0 else 0
        else:
            # If the error is tiny, don't push at all (Coast)
            return 1 if error > 0 else 0 

# --- SETUP ---
env = gym.make("CartPole-v1", render_mode="human")
observation, _ = env.reset()

# --- TUNING FOR MAC ---
# We use lower gains because the PWM logic is very efficient.
my_pid = PIDController(kp=0.3, ki=0.001, kd=0.8)

dt = 0.02
for _ in range(2000):
    angle = observation[2]
    
    action = my_pid.get_action(error=angle, dt=dt)
    
    observation, reward, terminated, truncated, _ = env.step(action)
    
    # Critical for Mac: Syncing the visual frame rate
    time.sleep(dt)

    if terminated or truncated:
        print("Final Stability Achieved.")
        break

env.close()