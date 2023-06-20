SOFTWARE DEPENDENCIES:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install gym==0.17.2
pip install nes_py
pip install gym_super_mario_bros
pip install atari_py
python -m atari_py.import_roms C:\Users\Staff\Downloads\Roms\ROMS\ROMS
pip install gym[atari,accept-rom-license]
pip install OpenCV-Python
pip install matplotlib

Optionally:
(1) Download Visual C++ Build Tools
(2) Install Development in C++
(3) Start Anaconda Prompt as administrator

EXAMPLE COMMANDS (for training and testing a DDQN agent):
python train_dqn_ale.py --env SuperMarioBros2-v0 --arch doubledqn --final-exploration-frames 200000 --steps 210000
python train_dqn_ale.py --env SuperMarioBros2-v0 --arch doubledqn --demo --load results\20230323T213356.460030-ddqn-supermario2-run1\2100000_finish --eval-n-runs 30 --render

EXAMPLE COMMANDS (for training and testing a DDQN agent with prioritised experience replay):
python train_dqn_ale.py --env SuperMarioBros2-v0 --arch doubledqn --prioritized --final-exploration-frames 2000000 --steps 2100000
python train_dqn_ale.py --env SuperMarioBros2-v0 --arch doubledqn --prioritized --demo --load results\20230323T213356.460030-ddqn-supermario2-run1\2100000_finish --eval-n-runs 30 --render

EXAMPLE COMMANDS (for training and testing a Rainbow agent):
python train_rainbow.py --env SuperMarioBros2-v0 --steps 2100000
python train_rainbow.py --env SuperMarioBros2-v0 --demo --load results\20230326T152344.634872\2100000_finish --render

NOTE: The folders above will depend on those existing on your PC, after being created during training.