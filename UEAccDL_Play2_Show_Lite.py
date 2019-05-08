from multiprocessing import Process, Queue
import threading
import numpy as np
from time import sleep
import airsim_local as airsim
import os
from UEAccDL import *
import keras

if __name__ == '__main__':
    # import tensorflow as tf
    # from keras.backend.tensorflow_backend import set_session

    # config = tf.ConfigProto(
    #     gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    #     # device_count = {'GPU': 1}
    # )
    # config=tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # session = tf.Session(config=config)
    # set_session(session)

    import matplotlib.pyplot as plt
    from matplotlib import patches
    import matplotlib.animation as animation

    # This import registers the 3D projection, but is otherwise unused.
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Path3DCollection

    # For 2D subplot
    from matplotlib.axes import Axes
    from matplotlib.image import AxesImage
    import matplotlib.lines as line
    from matplotlib.lines import Line2D
    from matplotlib.text import Text

#TODO check if API control is active, then go
def agent_process(queue: Queue):
    # Connect server
    # client = airsim.CarClient(ip='lucky7.twilightparadox.com')
    client = airsim.CarClient(ip='192.168.137.1')

    client.confirmConnection()
    client.enableApiControl(True)
    car_controls = airsim.CarControls()

    # Create Agent
    if os.path.isfile('UEADL_DQN_model.hdf5'):
        agent = Agent(model=keras.models.load_model('UEADL_DQN_model.hdf5', custom_objects={'huber_loss': huber_loss}),
                      explorer=LinearEpsilonAnnealingExplorer(0.000002, 0.000001, 100000))
    else:
        print('No model file.')
        agent = None

    # First preprocess of car_state, points_cloud, points_dist from client
    client.reset()
    sleep(1.5)
    _, points_cloud, points_dist = retrieve_raw_data(client)

    # Start running
    car_controls.throttle = 0.51  # Trained in 0.5, but runs greatly in 0.6
    client.setCarControls(car_controls)
    sleep(1.5)

    send_internal = 10
    step_counter = 0

    while True:
        # Convert frame before action
        frame_pre, grid_stack, input_state_stack = transform_frame(points_cloud, points_dist)

        # State_pre -> act to UE4
        action = agent.act(frame_pre)
        car_controls.steering = translate_action(action)
        client.setCarControls(car_controls)

        # Let it go for 10ms
        # TODO: remote lag is concerned so no need to sleep
        # sleep(0.01)

        # Get result: Update raw data after action
        collision_info, points_cloud_post, points_dist_post = retrieve_raw_data(client)

        # Calculate reward, terminal
        reward = calculate_reward(points_cloud_post, points_dist_post, collision_info)
        terminal = is_terminal(collision_info)

        # Send data to queue
        # TODO Solve memory leak: convert np.array -> list before sending
        # Prepare data
        if step_counter % send_internal == 0:
            queue.put_nowait((points_cloud.tolist(), grid_stack.tolist(), input_state_stack.tolist(),
                              car_controls.steering, reward))

        # If car crashed, episode finishes.
        if terminal:
            client.reset()
            car_controls.steering = translate_action(0)
            client.setCarControls(car_controls)
            sleep(1.5)
            _, points_cloud_post, points_dist_post = retrieve_raw_data(client)

        points_cloud, points_dist = points_cloud_post, points_dist_post

        # Data visualizing
        # print('AI step:', step_counter)
        step_counter += 1


def transform_frame(lidar_points_cloud, distance):
    """
    Transform raw LIDAR points cloud to sensor grid.
    Multi return version for realtime plot

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

    # Stack for graph 2
    grid_stack = np.append(grid, grid, 2)
    grid_stack = np.append(grid_stack, grid, 2)
    grid_stack = np.flipud(grid_stack)

    # Reduce process -> (32, 64, 1)
    processed_grid = resize(grid[...].astype(np.float32), (32, 64), preserve_range=True, anti_aliasing=True)
    processed_grid = exposure.adjust_gamma(processed_grid, 0.1)

    # Stack for graph 3
    input_state_stack = np.append(processed_grid, processed_grid, 2)
    input_state_stack = np.append(input_state_stack, processed_grid, 2)
    input_state_stack = np.flipud(input_state_stack)

    return np.squeeze(processed_grid, axis=2), grid_stack, input_state_stack


def translate_action(action=0):
    """
    Translate action number to steering degrees

    :param action:

    :return: Respective degrees
    """

    if action == 1:
        return -0.85
    elif action == 2:
        return -0.5
    elif action == 3:
        return 0.5
    elif action == 4:
        return 0.85
    else:
        return 0.0


if __name__ == '__main__':
    # Multiprocessing - AI process
    queue = Queue(maxsize=10000)  # communication tunnel between 2 processes
    p = Process(target=agent_process, args=[queue])
    p.daemon = True

    # Send dummy data for plotting for initializing the figure
    # queue.put((np.zeros((1, 3)), np.zeros((150, 300, 3)), np.zeros((32, 64, 3)), 0, 1, 0))
    p.start()

    # points_cloud, grid_stack, input_state_stack, direction, reward, step_counter = queue.get()
    grid_stack = np.zeros((150, 300, 3))
    points_cloud = np.zeros((1, 3))
    input_state_stack = np.zeros((32, 64, 3))
    direction, reward, step_counter = 0, 1, 0

    # Set Plots
    fig = plt.figure(figsize=(14, 3.5))

    # Graph 1: 3D scatter plot =============================================
    ax1 = fig.add_subplot(141, projection='3d')  # type:Axes3D
    ax1.set_xlim(0, 20)
    ax1.set_ylim(-15, 15)
    ax1.set_zlim(-0.6, 0.3)
    ax1.set_xlabel('X ( Back <---> Front )')
    ax1.set_ylabel('Y ( Right <---> Left )')
    ax1.set_zlabel('Z ( Bottom <---> Top )')
    sc = ax1.scatter(points_cloud[:, 0], np.negative(points_cloud[:, 1]),
                     np.negative(points_cloud[:, 2]))  # type:Path3DCollection

    # Graph 2: 2D image plot =============================================
    ax2 = fig.add_subplot(142)  # type:Axes
    ax2.set_title('Sensor Grid 150x300')
    im2 = ax2.imshow(np.zeros((150, 300, 3), dtype=np.uint8))  # type:AxesImage

    # Graph 3: Input state[2] plot =============================================
    ax3 = fig.add_subplot(143)  # type:Axes
    ax3.set_title('Input state[2] (32x64)')
    # Guide arts
    guide_side = ax3.scatter(np.asarray([11, 11, 11, 11, 11, 11, 11, 11, 11, 52, 52, 52, 52, 52, 52, 52, 52, 52]),
                             np.asarray([0, 4, 8, 12, 16, 20, 24, 28, 31, 0, 4, 8, 12, 16, 20, 24, 28, 31]), marker='2',
                             color='g')
    guide_range = ax3.add_artist(plt.Circle((31.55, 31), 11, color='g', fill=False))
    im3 = ax3.imshow(np.zeros((32, 64, 3), dtype=np.uint8))  # type:AxesImage

    # Graph 4: Summary Information plot =============================================
    ax4 = fig.add_subplot(144)  # type:Axes
    ax4.set_title('Indicators')
    # ax4.set_xlabel('X label here')
    # ax4.set_ylabel('Y label here')
    ax4.set_xlim(-1, 1)
    ax4.set_ylim(-1, 1.01)
    ax4.get_xaxis().set_visible(False)
    # ax4.axis('off')
    # step1 plot reward history
    xlim = ax4.get_xlim()
    xaxis = np.linspace(xlim[0], xlim[1], 100)
    reward_history = np.random.rand(100) * 0.5 * np.pi
    line_reward_his = ax4.plot(xaxis, reward_history, alpha=0.6)  # type: Line2D
    ax4.legend(['Reward History'], loc='lower right')
    ax4.add_line(line.Line2D(xlim, [0, 0], color='black'))

    # step2 plot direction
    ax4.add_patch(patches.Wedge((0, 0), r=1, theta1=0, theta2=180, fill=False, color='#be0119'))  # Scarlet
    # Guide points for directions
    ax4.scatter(np.asarray([-0.85, -0.5, 0.5, 0.85]),
                np.asarray([np.sqrt(1 - 0.7225), np.sqrt(1 - 0.25),
                            np.sqrt(1 - 0.25), np.sqrt(1 - 0.7225)]), marker='2', color='r')
    arrow_pointer = ax4.add_patch(patches.FancyArrow(0, 0, direction,
                                                     np.sqrt(1 - direction * direction),
                                                     length_includes_head=True, color='r', width=0.01,
                                                     head_width=0.1, head_length=0.4))  # type: patches.FancyArrow
    text_dir = ax4.text(-0.8, -0.25, 'Direction:' + str(direction), fontsize=18,
                        bbox={'facecolor': 'blue', 'alpha': 0.1, 'pad': 2})  # type: Text

    # step3 plot reward
    text_reward = ax4.text(-0.8, -0.55, 'Reward:' + "{0:.9f}".format(reward), fontsize=16,
                           bbox={'facecolor': 'red', 'alpha': 0.1, 'pad': 2})  # type: Text

    # Spacing adjust
    plt.tight_layout()
    print('Relocate figure window in 5 seconds.')
    plt.pause(5)

    local_counter = 0


    def plot_data(_):
        global points_cloud, grid_stack, input_state_stack, step_counter, \
            local_counter, reward_history, arrow_pointer, reward, direction, guide_range, guide_side

        # Update graph 1
        # TODO what if no 3D?
        sc._offsets3d = (points_cloud[:, 0], np.negative(points_cloud[:, 1]), np.negative(points_cloud[:, 2]))
        # Update graph 2
        im2.set_data(grid_stack.astype(np.float32))
        # Update graph 3
        im3.set_data(input_state_stack.astype(np.float32))
        if reward < 1:
            guide_side.set_color('r')
            guide_range.set_color('r')
        else:
            guide_side.set_color('#89fe05')  # Lime green
            guide_range.set_color('#89fe05')

        # Update graph 4
        # reward_history
        reward_history = np.append(reward_history, reward)
        if reward_history.size > 100:
            reward_history = np.delete(reward_history, [0])
        line_reward_his[0].set_ydata(reward_history)
        # Direction
        arrow_pointer.remove()
        arrow_pointer = ax4.add_patch(patches.FancyArrow(0, 0, direction, np.sqrt(1 - direction * direction),
                                                         length_includes_head=True, color='r', width=0.01,
                                                         head_width=0.1, head_length=0.4))
        # # Text direction
        text_dir.set_text('Direction:' + str(direction))
        # # Text reward
        text_reward.set_text('Reward:' + "{0:.9f}".format(reward))

        local_counter += 1
        print('=-----------------------------------------------')


    def update_data():
        global points_cloud, grid_stack, input_state_stack, direction, reward, step_counter

        while True:
            if not queue.empty():
                # TODO Convert back: list -> np.array
                points_cloud_list, grid_stack_list, input_state_stack_list, direction, reward = queue.get()
                points_cloud = np.asarray(points_cloud_list, dtype='float32')
                grid_stack = np.asarray(grid_stack_list, dtype='float32')
                input_state_stack = np.asarray(input_state_stack_list, dtype='float32')
                print('Data retrieved. <<<<<<<<<<\n')
            else:
                # print('Queue empty.\n')
                pass
            sleep(0.075)


    # Daemon data retrieving thread
    t = threading.Thread(target=update_data, name='Thread-Receiver')
    t.daemon = True
    t.start()

    ani = animation.FuncAnimation(fig, plot_data, interval=200, repeat=True)
    plt.show()

    # Connect server
    client = airsim.CarClient(ip='lucky7.twilightparadox.com')
    client.confirmConnection()
    client.enableApiControl(True)
    car_controls = airsim.CarControls()

    def manual_override(_manual):
        client.enableApiControl(not _manual)

    def set_throttle(_s):
        car_controls.throttle = _s
        client.setCarControls(car_controls)
