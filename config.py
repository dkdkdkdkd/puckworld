
class Config:
    # ENV = 'Snake-v0'
    ENV = 'PuckWorld-v0'
    GAMMA = 0.99

    MAX_STEPS = 10000000
    LEARNING_START = 50000
    BATCH_SIZE = 32
    CAPACITY = 1000000
    TARGET_UPDATE_FREQ = 10000

    EPSILON_INIT = 1.0
    EPSILON_MIN = 0.1

