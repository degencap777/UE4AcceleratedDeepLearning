import os
from time import sleep
import airsim_local as airsim
import numpy as np
from skimage import exposure
from skimage.transform import resize
import keras
from keras import backend as K
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Path3DCollection
from matplotlib.lines import Line2D


class ReplayMemory(object):
    """
    ReplayMemory keeps track of the environment dynamic.
    We store all the transitions (s(t), action, s(t+1), reward, done).
    The replay memory allows us to efficiently sample minibatches from it, and generate the correct state representation
    (w.r.t the number of previous frames needed).

    States stored as np.float32
    """

    def __init__(self, size, sample_shape, short_term_mem_stack=3):
        self._pos = 0
        self._count = 0
        self._max_size = size
        self._short_term_mem_stack = max(1, short_term_mem_stack)
        self._state_shape = sample_shape
        self._states = np.zeros((size,) + sample_shape, dtype=np.float32)  # value of state(32x64)
        self._actions = np.zeros(size, dtype=np.uint8)
        self._rewards = np.zeros(size, dtype=np.float32)
        self._terminals = np.zeros(size, dtype=np.bool)

    def __len__(self):
        """ Returns the number of items currently present in the memory
        Returns: Int >= 0
        """
        return self._count

    def append(self, frame: np.ndarray, action, reward, terminal):
        """ Appends the specified transition to the memory.

        Attributes:
            frame (Tensor[sample_shape]): The state to append
            action (int): An integer representing the action done
            reward (float): An integer representing the reward received for doing this action
            terminal (bool): A boolean specifying if this state is a terminal (episode has finished)
        """
        assert frame.shape == self._state_shape, \
            'Invalid state shape (required: %s, got: %s)' % (self._state_shape, frame.shape)

        self._states[self._pos] = frame
        self._actions[self._pos] = action
        self._rewards[self._pos] = reward
        self._terminals[self._pos] = terminal

        self._count = max(self._count, self._pos + 1)
        self._pos = (self._pos + 1) % self._max_size

    def sample(self, size):
        """ Generate size random integers mapping indices in the memory.
            The returned indices can be retrieved using #get_state().
            See the method #minibatch() if you want to retrieve samples directly.

        Attributes:
            size (int): The minibatch size

        Returns:
             Indexes of the sampled states ([int])
        """

        # Local variable access is faster in loops
        count, pos, history_len, terminals = self._count - 1, self._pos, \
                                             self._short_term_mem_stack, self._terminals
        indexes = []

        while len(indexes) < size:
            index = np.random.randint(history_len, count)

            if index not in indexes:

                # if not wrapping over current pointer,
                # then check if there is terminal state wrapped inside
                if not (index >= pos > index - history_len):
                    if not terminals[(index - history_len):index].any():
                        indexes.append(index)

        return indexes

    def minibatch(self, size):
        """ Generate a minibatch with the number of samples specified by the size parameter.

        Attributes:
            size (int): Minibatch size

        Returns:
            tuple: Tensor[minibatch_size, input_shape...], [int], [float], [bool]
        """
        indexes = self.sample(size)

        states_pre = np.moveaxis(
            np.array([self.get_state(index) for index in indexes]), 1, -1)
        states_post = np.moveaxis(
            np.array([self.get_state(index + 1) for index in indexes]), 1, -1)
        actions = self._actions[indexes]
        rewards = self._rewards[indexes]
        terminals = self._terminals[indexes]

        return states_pre, actions, states_post, rewards, terminals

    def get_state(self, index):
        """
        Return the specified state with the replay memory. A state consists of
        the last `stack_length` perceptions.

        Attributes:
            index (int): State's index

        Returns:
            State at specified index (Tensor[stack_length, input_shape...])
        """
        if self._count == 0:
            raise IndexError('Empty Memory')

        index %= self._count
        stack_length = self._short_term_mem_stack

        # If index > stack_length, take from a slice
        if index >= stack_length:
            return self._states[(index - (stack_length - 1)):index + 1, ...]
        else:
            indexes = np.arange(index - stack_length + 1, index + 1)
            return self._states.take(indexes, mode='wrap', axis=0)


class ShortTermMemory(object):
    """
    Accumulator keeping track of the N previous frames to be used by the agent
    for evaluation
    """

    def __init__(self, shape):
        self._buffer = np.zeros(shape, dtype=np.float32)

    @property
    def value(self):
        """ Underlying buffer with N previous states stacked along first axis

        Returns:
            Tensor[shape]
        """
        return self._buffer

    def append(self, state):
        """ Append state to the history

        Attributes:
            state (Tensor) : The state to append to the memory
        """
        self._buffer[:, :, :-1] = self._buffer[:, :, 1:]
        self._buffer[:, :, -1] = state

    def reset(self):
        """ Reset the memory. Underlying buffer set all indexes to 0

        """
        self._buffer.fill(0)


class LinearEpsilonAnnealingExplorer(object):
    """
    Exploration policy using Linear Epsilon Greedy

    Attributes:
        start (float): start value
        end (float): end value
        steps (int): number of steps between start and end
    """

    def __init__(self, start, end, steps):
        self._start = start
        self._stop = end
        self._steps = steps

        self._step_size = (end - start) / steps

    def __call__(self, num_actions=5):
        """
        Select a random action out of `num_actions` possibilities.

        Attributes:
            num_actions (int): Number of actions available
        """
        return np.random.choice(num_actions)

    def _epsilon(self, step):
        """ Compute the epsilon parameter according to the specified step

        Attributes:
            step (int)
        """
        if step < 0:
            return self._start
        elif step > self._steps:
            return self._stop
        else:
            return self._step_size * step + self._start

    def is_exploring(self, step):
        """ Commodity method indicating if the agent should explore

        Attributes:
            step (int) : Current step

        Returns:
             bool : True if exploring, False otherwise
        """
        return np.random.rand() < self._epsilon(step)


def huber_loss(y, y_estimate, delta=1, in_keras=True):
    """
    Compute the Huber Loss as part of the model

    Huber Loss is more robust to outliers. It is defined as:
     if |y - y_estimate| < delta :
        0.5 * (y - y_estimate)**2
    else :
        delta * |y - y_estimate| - 0.5 * delta**2

    :param y: Target value
    :param y_estimate: Estimated value
    :param delta: Outliers threshold
    :param in_keras: If in keras mode

    :return: Loss value
    """

    error = y - y_estimate
    error_abs = abs(error)

    quadratic_term = 0.5 * np.square(error)
    linear_term = (delta * error_abs) - 0.5 * delta * delta

    use_linear_term = error_abs > 1.0  # Vectorized 'if'
    if in_keras:
        # Explicitly cast the booleans to floats
        use_linear_term = K.cast(use_linear_term, 'float32')
    return use_linear_term * linear_term + (1 - use_linear_term) * quadratic_term


def transform_frame(lidar_points_cloud, distance):
    """
    Transform raw LIDAR points cloud to sensor grid.

    :param distance: Distance in meter
    :param lidar_points_cloud: Raw LIDAR points cloud, coordinates in decimeter

    :return: Array of lidar points grid: shape(32, 64, 1) -> squeeze to (32, 64)
    """

    # Convert grid: 30m x 15m -> 300dm x 150dm
    _points = lidar_points_cloud[:, :2]  # Squeeze to 2D
    _points = (_points * 10).astype(np.uint16)  # Convert to decimeter
    _points = _points[distance <= 15]  # Choose points which have distance less than 15m
    _points[:, 1] += 150  # Shift horizontal
    grid = np.zeros((150, 300, 1), dtype=np.uint16)  # Create sensor grid to receive lidar_points
    for point in _points:  # Project points to grid
        # TODO check 'index 65535 is out of bounds for axis 0 with size 150'
        if point[0] <= 150 and point[1] <= 300:
            grid[point[0], point[1], 0] = 1

    # Reduce process -> (32, 64, 1)
    processed_grid = resize(grid[...].astype(np.float32), (32, 64), preserve_range=True, anti_aliasing=True)
    processed_grid = exposure.adjust_gamma(processed_grid, 0.1)
    # for p in np.nditer(processed_grid, op_flags=['readwrite']):  # Binarize reduced grid
    #     if p != 0:
    #         p[...] = 1

    return np.squeeze(processed_grid, axis=2)


def is_terminal(collision_info: airsim.CollisionInfo):
    """
    Terminal function.

    :param collision_info:

    :return: If the state is terminal
    """

    return collision_info.has_collided


def calculate_reward(lidar_points, distance, collision_info: airsim.CollisionInfo):
    """
    Calculate reward.

    :param lidar_points:
    :param distance:
    :param collision_info:

    :return: Reward of this state
    """

    frame = transform_frame(lidar_points, distance)  # type: np.ndarray

    # Close distance alert term
    alert_range = 6.0
    alert_term = -1 * lidar_points[distance <= alert_range, :].size / lidar_points.size

    if alert_term == 0:
        alert_term = 1

        # Append two side-checks of Shape(32,64) -> (32, 10) * 2
        side_points = np.append(frame[:, :12], frame[:, -12:])
        side_term = -1 * np.count_nonzero(side_points) / side_points.size
        if side_term == 0:
            side_term = 1
    else:
        side_term = -1

    # Final reward formula
    # Shift reward to [0, 1] with weights of 50/50 for alert and side term respectively
    reward = alert_term * 0.25 + side_term * 0.25 + 0.5

    if collision_info.has_collided:
        reward = -5

    # Debug
    print('Side:', side_term, 'Alert:', alert_term, 'Reward:', reward)

    return reward


def translate_action(action=0):
    """
    Translate action number to steering degrees

    :param action:

    :return: Respective degrees
    """

    if action == 1:
        return -0.8
    elif action == 2:
        return -0.5
    elif action == 3:
        return 0.5
    elif action == 4:
        return 0.8
    else:
        return 0.0


class Agent(object):
    """
    Agent Introduction
    """

    def __init__(self, model=None, model_target=None, input_shape=(32, 64, 3), n_actions=5,
                 gamma=0.99, explorer=LinearEpsilonAnnealingExplorer(0.0020, 0.0005, 100000),
                 learning_rate=0.00025, mini_batch_size=32,
                 memory_size=1000000, train_after=100000, train_interval=3,  # trainAfter 150k, interval 3
                 target_update_interval=10000, trainings_per_epoch=20000,
                 eval_states_batch_size=10000):  # train_after must bigger than 1.5 * eval batch size
        self.model = model  # type:keras.engine.training.Model
        self.model_target = model_target  # type:keras.engine.training.Model

        self.input_shape = input_shape
        self.n_actions = n_actions
        self.gamma = gamma

        self._train_after = train_after
        self._train_interval = train_interval
        self._target_update_interval = target_update_interval
        self._learning_rate = learning_rate
        self._explorer = explorer
        self._mini_batch_size = mini_batch_size
        self._short_term_memory = ShortTermMemory(input_shape)
        self._memory = ReplayMemory(memory_size, input_shape[:-1], input_shape[-1])  # Long term memory
        self._num_actions_taken = 0  # Action tracker

        # For evaluation
        self._is_eval_states_ready = False
        self._eval_states_batch = None
        self._eval_states_batch_size = eval_states_batch_size
        self._trainings_per_epoch = trainings_per_epoch
        self._num_trainings_done = 0  # Training tracker
        self._num_epochs_gone = 0
        self._q_average_evaluations = []  # Value of var[7] indicates the eval of Epoch 8
        self._evaluation_cool_down = 0

    def create_model(self):
        """
        Create model
        """

        # Input layers
        state_input = keras.layers.Input(self.input_shape, name='states')
        actions_input = keras.layers.Input((self.n_actions,), name='mask')
        # Hidden layers
        conv_1 = keras.layers.convolutional.Conv2D(24, (4, 4), activation='relu', strides=(2, 2))(state_input)
        pool1 = keras.layers.MaxPooling2D(padding='same')(conv_1)
        conv_2 = keras.layers.convolutional.Conv2D(32, (3, 3), activation='relu', strides=(1, 1))(pool1)
        conv_flattened = keras.layers.core.Flatten()(conv_2)
        hidden = keras.layers.Dense(512, activation='relu')(conv_flattened)
        # Output layer
        output = keras.layers.Dense(self.n_actions)(hidden)
        filtered_output = keras.layers.multiply([output, actions_input])

        # optimizer = keras.optimizers.RMSprop(lr=0.00025, rho=0.95, epsilon=0.01)
        # Adam based SGD
        optimizer = keras.optimizers.Adam(lr=self._learning_rate, epsilon=0.01)

        self.model = keras.models.Model(inputs=[state_input, actions_input], outputs=filtered_output)

        self.model.compile(optimizer, loss=huber_loss)
        self.model.summary()

        self.model.save('UEADL_DQN_model_new.hdf5')
        self.model_target = keras.models.load_model('UEADL_DQN_model_new.hdf5',
                                                    custom_objects={'huber_loss': huber_loss})

    def act(self, frame: np.ndarray):
        """
        Acts
        """

        self._short_term_memory.append(frame)

        # Explore or estimate
        if self._explorer.is_exploring(self._num_actions_taken):
            action = self._explorer(self.n_actions)
        else:
            state = self._short_term_memory.value  # np.float32
            q_val_all = self.model.predict([[state], [np.ones(self.n_actions)]])
            action = np.argmax(q_val_all)
            # print('Q Predicts:', q_val_all)
        self._num_actions_taken += 1

        return action

    def observe(self, frame, action, reward, terminal):
        """
        Observe frame, memorize.

        :param frame:
        :param action:
        :param reward:
        :param terminal:
        :return:
        """

        # Trace things of episode
        if terminal:
            self._short_term_memory.reset()
        # Append to long term memory
        self._memory.append(frame, action, reward, terminal)

    def train(self):
        """
        Train process.

        """

        agent_step = self._num_actions_taken  # Local var is faster

        # Train process
        if agent_step >= self._train_after:
            if (agent_step % self._train_interval) == 0:
                # Sample a mini batch
                states_pre, actions, states_post, rewards, terminals = self._memory.minibatch(self._mini_batch_size)
                # One-hot encoding
                actions_one_hot = np.zeros((self._mini_batch_size, self.n_actions), dtype='int8')
                for idx in range(self._mini_batch_size):
                    actions_one_hot[idx][actions[idx]] = 1

                next_q_values = self.model_target.predict_on_batch([states_post,
                                                                    np.ones((self._mini_batch_size, self.n_actions))])
                next_q_values[terminals] = 0  # Set all actions' q value to 0 if it is terminal.(np broadcast)
                q_values = rewards + self.gamma * np.max(next_q_values, axis=1)  # Formula
                self.model.train_on_batch([states_pre, actions_one_hot], q_values[:, None] * actions_one_hot)

                # Track training times
                self._num_trainings_done += 1

            # Save model and Update model_target periodically
            if (agent_step % self._target_update_interval) == 0:
                print('Saving model as UEADL_DQN_model.hdf5')
                self.model.save('UEADL_DQN_model.hdf5')
                self.model_target = keras.models.load_model('UEADL_DQN_model.hdf5',
                                                            custom_objects={'huber_loss': huber_loss})
        # Set fixed states set for evaluation
        elif agent_step > self._train_after - 100 and not self._is_eval_states_ready:
            self._eval_states_batch, _, _, _, _ = self._memory.minibatch(self._eval_states_batch_size)
            self._is_eval_states_ready = True

    def track_performance(self):
        """
        Data visualization. matplotlib
        Evaluate and plot on every epoch end.

        :return: True when new evaluation added, else False.

        """

        self._num_epochs_gone, remainder = divmod(self._num_trainings_done, self._trainings_per_epoch)

        if self._evaluation_cool_down == 0 and self._is_eval_states_ready and \
                self._num_epochs_gone > 0 and remainder == 0:  # At the timing that epoch increases 1
            result = self.model.predict([self._eval_states_batch,
                                         np.ones(
                                             (self._eval_states_batch_size, self.n_actions))])  # result shape(10k, 5)
            q_avg = np.average(np.max(result, axis=1))  # Final average Q as result of this epoch
            self._q_average_evaluations.append(q_avg)

            # Log to file
            f = open('QavgLog.txt', 'a')
            f.write(str(self._num_epochs_gone) + ',\t' + str(q_avg) + '\n')
            f.close()

            print('Epoch ' + str(self._num_epochs_gone) + ' Evaluations updated.')
            self._evaluation_cool_down = 3
            return True
        elif self._evaluation_cool_down > 0:
            self._evaluation_cool_down -= 1

        return False

    @property
    def q_average_evaluations(self):
        """
        Getter method

        :return: q_average_evaluations
        """

        return self._q_average_evaluations

    def plot_real_time_status(self):
        """
        Realtime plot for presentation.

        :return:
        """
        if self._num_actions_taken % 10 == 0:
            print('Actions:', self._num_actions_taken, 'Trainings:', self._num_trainings_done, 'Epochs:',
                  self._num_epochs_gone)


def retrieve_raw_data(client: airsim.CarClient):
    collision_info = client.simGetCollisionInfo()
    points_cloud = np.array(client.getLidarData(lidar_name='Lidar1').point_cloud, dtype=np.dtype('f4'))
    points_cloud = np.reshape(points_cloud, (int(points_cloud.shape[0] / 3), 3))
    points_dist = np.linalg.norm(points_cloud, axis=1)

    return collision_info, points_cloud, points_dist


if __name__ == '__main__':

    EPISODE = 0

    # Set plot figure
    # fig = plt.figure()
    # plt.pause(0.01)
    # q_avg_axis = fig.add_subplot(111)
    # line, = q_avg_axis.plot(evals_data)  # type: Line2D

    # def plot_data(_):
    #
    #     pass
    # ani = animation.FuncAnimation(fig, plot_data, repeat=True)

    # Connect server
    client = airsim.CarClient(ip='127.0.0.1')
    client.confirmConnection()
    client.enableApiControl(True)
    car_controls = airsim.CarControls()

    # Create Agent
    if os.path.isfile('UEADL_DQN_model.hdf5'):
        agent = Agent(model=keras.models.load_model('UEADL_DQN_model.hdf5', custom_objects={'huber_loss': huber_loss}),
                      model_target=keras.models.load_model('UEADL_DQN_model.hdf5',
                                                           custom_objects={'huber_loss': huber_loss}))
    else:
        agent = Agent()
        agent.create_model()
    print(agent.model.metrics_names)

    client.reset()
    sleep(1.5)
    
    # First preprocess of car_state, points_cloud, points_dist from client
    _, points_cloud, points_dist = retrieve_raw_data(client)

    car_controls.throttle = 0.65  # Trained in 0.5, but runs greatly in 0.65
    client.setCarControls(car_controls)
    sleep(1.5)

    while True:
        # Convert frame before action
        frame_pre = transform_frame(points_cloud, points_dist)

        # State_pre -> act to UE
        action = agent.act(frame_pre)
        car_controls.steering = translate_action(action)
        client.setCarControls(car_controls)

        # Let it go for 10ms
        sleep(0.01)

        # Get result: Update raw data after action
        collision_info, points_cloud_post, points_dist_post = retrieve_raw_data(client)

        # Calculate reward, terminal
        reward = calculate_reward(points_cloud_post, points_dist_post, collision_info)
        terminal = is_terminal(collision_info)

        # Observe
        agent.observe(frame_pre, action, reward, terminal)

        # Train agent
        agent.train()

        # If car crashed, episode finishes.
        if terminal:
            client.reset()
            car_controls.steering = translate_action(0)
            client.setCarControls(car_controls)
            sleep(1.5)
            EPISODE += 1
            # TODO Reset posts too, to not pass 'the last moment' of last episode to the first state of new episode
            _, points_cloud_post, points_dist_post = retrieve_raw_data(client)

        points_cloud, points_dist = points_cloud_post, points_dist_post

        # Data visualizing

        agent.plot_real_time_status()

        # Update data for plotting
        if agent.track_performance():
            # q_avg_axis.clear()
            # q_avg_axis.set_xlabel('Epochs')
            # q_avg_axis.set_ylabel('Average Q (action value)')
            # q_avg_axis.plot(agent.q_average_evaluations)
            plt.gca().clear()
            plt.xlabel('Epochs')
            plt.ylabel('Average Q (action value)')
            plt.plot(agent.q_average_evaluations)
            plt.savefig('Performance Q Average vs Epochs', dpi=200, transparent=True)  # dpi=200 -> 1280x960
            plt.pause(0.01)
