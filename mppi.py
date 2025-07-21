import ur5_env as env

ur5 = env.ur5(render_mode="human")
ur5.reset()

# iterable that has a next method that can run x times
for step in range(100):
    obs, reward, done, term, info = ur5.step(ur5.action_space.sample())
