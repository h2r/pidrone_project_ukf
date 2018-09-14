#!/usr/bin/env python

# ROS imports
import rospy
from sensor_msgs.msg import Imu, Range
from pidrone_pkg.msg import State

# UKF imports
# The matplotlib imports and the matplotlib.use('Pdf') line make it so that the
# UKF code that imports matplotlib does not complain. Essentially, the
# use('Pdf') call allows a plot to be created without a window (allows it to run
# through ssh)
import matplotlib
matplotlib.use('Pdf')
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.kalman import MerweScaledSigmaPoints

# Other imports
import numpy as np
import argparse
import os


class UKFStateEstimator2D(object):
    """
    Class that estimates the state of the drone using an Unscented Kalman Filter
    (UKF) applied to raw sensor data. The filter only tracks the quadcopter's
    motion along one spatial dimension: the global frame z-axis. In this
    simplified case, we assume that the drone's body frame orientation is
    aligned with the world frame (i.e., no roll, pitch, or yaw), and that it is
    only offset along the z-axis.
    """
    
    def __init__(self, ir_throttled=False, imu_throttled=False):
        self.ir_topic_str = '/pidrone/infrared'
        self.imu_topic_str = '/pidrone/imu'
        throttle_suffix = '_throttle'
        
        if ir_throttled:
            self.ir_topic_str += throttle_suffix
        if imu_throttled:
            self.imu_topic_str += throttle_suffix
            
        self.in_callback = False

        self.initialize_ukf()
        
        # The last time that we received an input and formed a prediction with
        # the state transition function
        self.last_state_transition_time = None
        
        # Time in seconds between consecutive state transitions, dictated by
        # when the inputs come in
        self.dt = None
        
        # Initialize the last control input as 0 m/s^2 along the z-axis
        self.last_control_input = np.array([0.0])
        
        self.initialize_ros()
        
    def initialize_ros(self):
        """
        Initialize ROS-related objects, e.g., the node, subscribers, etc.
        """
        self.node_name = os.path.splitext(os.path.basename(__file__))[0]
        print 'Initializing {} node...'.format(self.node_name)
        rospy.init_node(self.node_name)
        
        # Subscribe to topics to which the drone publishes in order to get raw
        # data from sensors, which we can then filter
        rospy.Subscriber(self.imu_topic_str, Imu, self.imu_data_callback)
        rospy.Subscriber(self.ir_topic_str, Range, self.ir_data_callback)
        
        # Create the publisher to publish state estimates
        self.state_pub = rospy.Publisher('/pidrone/state/ukf_2d', State, queue_size=1,
                                         tcp_nodelay=False)
        
    def initialize_ukf(self):
        """
        Initialize the parameters of the Unscented Kalman Filter (UKF) that is
        used to estimate the state of the drone.
        """
        
        # Number of state variables being tracked
        self.state_vector_dim = 2
        # The state vector consists of the following column vector.
        # Note that FilterPy initializes the state vector with zeros.
        # [[z],
        #  [z_vel]]
        
        # Number of measurement variables that the drone receives
        self.measurement_vector_dim = 1
        # The measurement variables consist of the following vector:
        # [[slant_range]]
        
        # Object to generate sigma points for the UKF
        sigma_points = MerweScaledSigmaPoints(n=self.state_vector_dim,
                                              alpha=0.1,
                                              beta=2.0,
                                              kappa=(3.0-self.state_vector_dim))
                                              
        # Create the UKF object
        # Note that dt will get updated dynamically as sensor data comes in,
        # as will the measurement function, since measurements come in at
        # distinct rates. Setting compute_log_likelihood=False saves some
        # computation.
        self.ukf = UnscentedKalmanFilter(dim_x=self.state_vector_dim,
                                         dim_z=self.measurement_vector_dim,
                                         dt=1.0,
                                         hx=self.measurement_function,
                                         fx=self.state_transition_function,
                                         points=sigma_points,
                                         compute_log_likelihood=False)
        self.initialize_ukf_matrices()

    def initialize_ukf_matrices(self):
        """
        Initialize the covariance matrices of the UKF
        """
        # Initialize state covariance matrix P, to be updated in a callback:
        self.ukf.P = np.diag([0.1, 0.2])
        
        # Initialize the process noise covariance matrix Q:
        self.ukf.Q = np.diag([0.01, 1.0])*0.0005
        
        # TODO: Initialize the measurement covariance matrix R, containing IR
        #       range variance (units: m^2) determined experimentally in a
        #       static setup
        # self.ukf.R = np.array([?])
        
    def initialize_input_time(self, msg):
        """
        Initialize the input time (self.last_state_transition_time) based on the
        timestamp in the header of a ROS message. This is called before we start
        filtering in order to attain an initial time value, which enables us to
        then compute a time interval self.dt later on.
        
        msg : a ROS message that includes a header with a timestamp
        """
        self.last_time_secs = msg.header.stamp.secs
        self.last_time_nsecs = msg.header.stamp.nsecs
        self.last_state_transition_time = (self.last_time_secs +
                                           self.last_time_nsecs*1e-9)
        
    def imu_data_callback(self, data):
        """
        Handle the receipt of an Imu message. Only take the linear acceleration
        along the z-axis.
        """
        if self.in_callback:
            return
        self.in_callback = True
        ##########################################
        # TODO: Implement this method to handle the control input from the IMU
        
        ##########################################
        self.in_callback = False
                        
    def ir_data_callback(self, data):
        """
        Handle the receipt of a Range message from the IR sensor, forming both a
        PREDICTION and a MEASUREMENT UPDATE.
        """
        if self.in_callback:
            return
        self.in_callback = True
        ##########################################
        # TODO: Implement the prediction and update steps upon receipt of a
        #       measurement from the IR sensor
        
        ##########################################
        self.in_callback = False
                        
    def publish_current_state(self):
        """
        Publish the current state estimate and covariance from the UKF. This is
        a State message containing:
            - Header
            - PoseWithCovariance
            - TwistWithCovariance
        Note that a lot of these ROS message fields will be left empty, as the
        2D UKF does not track information on the entire state space of the
        drone.
        """
        state_msg = State()
        state_msg.header.stamp.secs = self.last_time_secs
        state_msg.header.stamp.nsecs = self.last_time_nsecs
        state_msg.header.frame_id = 'global'
        
        # Get the current state estimate from self.ukf.x
        state_msg.pose_with_covariance.pose.position.z = self.ukf.x[0]
        state_msg.twist_with_covariance.twist.linear.z = self.ukf.x[1]
        
        # Fill the rest of the message with NaN
        state_msg.pose_with_covariance.pose.position.x = np.nan
        state_msg.pose_with_covariance.pose.position.y = np.nan
        state_msg.pose_with_covariance.pose.orientation.x = np.nan
        state_msg.pose_with_covariance.pose.orientation.y = np.nan
        state_msg.pose_with_covariance.pose.orientation.z = np.nan
        state_msg.pose_with_covariance.pose.orientation.w = np.nan
        state_msg.twist_with_covariance.twist.linear.x = np.nan
        state_msg.twist_with_covariance.twist.linear.y = np.nan
        state_msg.twist_with_covariance.twist.angular.x = np.nan
        state_msg.twist_with_covariance.twist.angular.y = np.nan
        state_msg.twist_with_covariance.twist.angular.z = np.nan
        
        # Prepare covariance matrices
        # 36-element array, in a row-major order, according to ROS msg docs
        pose_cov_mat = np.full((36,), np.nan)
        twist_cov_mat = np.full((36,), np.nan)
        pose_cov_mat[14] = self.ukf.P[0, 0]  # z variance
        twist_cov_mat[14] = self.ukf.P[1, 1]  # z velocity variance
        
        # Add covariances to message
        state_msg.pose_with_covariance.covariance = pose_cov_mat
        state_msg.twist_with_covariance.covariance = twist_cov_mat
        
        self.state_pub.publish(state_msg)

    def state_transition_function(self, x, dt, u):
        """
        The state transition function to compute the prior in the prediction
        step, propagating the state to the next time step.
        
        x : current state. A NumPy array
        dt : time step. A float
        u : control input. A NumPy array
        
        returns: A NumPy array that is the prior state estimate. Array
                 dimensions must be the same as the state vector x.
        """
        # TODO: Implement this method, following the math that you derived.
        pass
        
    def measurement_function(self, x):
        """
        Transform the state x into measurement space. In this simple model, we
        assume that the range measurement corresponds exactly to position along
        the z-axis, as we are assuming there is no pitch and roll.
        
        x : current state. A NumPy array
        
        returns: A NumPy array of the state vector transformed into measurement
                 space. Array dimensions must be the same as the measurement
                 vector.
        """
        # TODO: Implement this method, following the math that you derived.
        pass
        
        
def main():
    parser = argparse.ArgumentParser(description=('Estimate the drone\'s state '
                                     'with a UKF in one spatial dimension'))
    # Arguments to determine if the throttle command is being used. E.g.:
    #   rosrun topic_tools throttle messages /pidrone/infrared 40.0
    parser.add_argument('--ir_throttled', action='store_true',
            help=('Use throttled infrared topic /pidrone/infrared_throttle'))
    parser.add_argument('--imu_throttled', action='store_true',
            help=('Use throttled IMU topic /pidrone/imu_throttle'))
    args = parser.parse_args()
    se = UKFStateEstimator2D(ir_throttled=args.ir_throttled,
                             imu_throttled=args.imu_throttled)
    try:
        # Wait until node is halted
        rospy.spin()
    finally:
        # Upon termination of this script, print out a helpful message
        print '{} node terminating.'.format(se.node_name)
        print 'Most recent state vector:'
        print se.ukf.x
        
if __name__ == '__main__':
    main()
