from ale_py import ALEInterface, roms
import matplotlib.pyplot as plt

ale = ALEInterface()
ale.loadROM(roms.get_rom_path("breakout"))
ale.reset_game()

reward = ale.act(0)  # noop
screen_obs = ale.getScreenRGB()

plt.imshow(screen_obs)
plt.show()