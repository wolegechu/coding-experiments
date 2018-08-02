#!/usr/bin/env python
# encoding:utf-8
# Python libs
import sys, time

# numpy and math
import numpy as np
from math import radians, copysign, sqrt, pow, pi, atan2,tan

# OpenCV
import cv2

# Ros libraries
import roslib
import rospy
import tf
from tf.transformations import euler_from_quaternion

# Ros Messages
from geometry_msgs.msg import Twist, Point, Quaternion
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Imu

# darknet
import darknet as dn

# utils
from detector import coordinate_calculate, location_detect




TURN_BALL_ANGULAR_Z = 0.1
FIND_BALL_ANGULAR_Z = 1
class mc_detect_ctrl:

    def __init__(self):
        self._N_FRAME = 5
        # init ros node
        rospy.init_node('mc_detect_ctrl', anonymous=False)

        # publish processed image
        self.image_pub = rospy.Publisher("/mc_cam_detect/image/compressed",
            CompressedImage,  queue_size = 1)
        # publish map
        self.map_pub = rospy.Publisher("/map/image/compressed",
            CompressedImage,  queue_size = 1)
        # publish v cmd
        self.cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        
        # load net and meta to darknet
        self.net = dn.load_net(b"/home/momenta/catkin_ws/src/mc_frenta/mc_detect/nodes/yolov3-tiny.cfg",
         b"/home/momenta/catkin_ws/src/mc_frenta/mc_detect/nodes/yolov3-tiny_9700000.weights", 0)
        self.meta = dn.load_meta(b"/home/momenta/catkin_ws/src/mc_frenta/mc_detect/nodes/mc.data")
        self.all_classes = ['door', 'robot', 'football']
        self.box_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

        # get odom
        self.tf_listener = tf.TransformListener()
        self.odom_frame = 'odom'

        try:
            self.tf_listener.waitForTransform(self.odom_frame, 'base_footprint', rospy.Time(), rospy.Duration(1.0))
            self.base_frame = 'base_footprint'
        except (tf.Exception, tf.ConnectivityException, tf.LookupException):
            try:
                self.tf_listener.waitForTransform(self.odom_frame, 'base_link', rospy.Time(), rospy.Duration(1.0))
                self.base_frame = 'base_link'
            except (tf.Exception, tf.ConnectivityException, tf.LookupException):
                rospy.loginfo("Cannot find transform between odom and base_link or base_footprint")
                rospy.signal_shutdown("tf Exception")

        # subscribe imu
        rospy.Subscriber("/imu",Imu, self.imu_sub,  queue_size = 1)
        
        # subscribe image
        rospy.Subscriber("/mc_cam_pub/image/compressed",
            CompressedImage, self.img_sub, queue_size=1)
        
        # init flag and raw value
        self.start = False
        self.ready = True
        self.found = False
        self.ktheta = pi/4
        self.ktheta_tiny = pi/6
        self.turn_flag = True
        self.yaw = 0
        self.pre_odom = (Point(0.23,0.4,0), 0)
        self.map = cv2.imread('/home/momenta/catkin_ws/src/mc_frenta/mc_detect/nodes/map.jpg')
        ## final
        self.previous_coordinate = previous_coordinate = {'robot': np.array((0.23,0.40)), 'ball': np.array((1.45,0.85)),
                                                        'enemy_robot': np.array((2.75,1.55))}
        self.coordinate = previous_coordinate = {'robot': np.array((0.23,0.40)), 'ball': np.array((1.45,0.85)),
                                               'enemy_robot': np.array((2.75,1.55))}
        self.location = {'contain_ball':0}

        self.his_loca = []
        self.his_odom = []
        self.his_coordinate = []
        
        self.ctrl()
        
    def imu_sub(self, imu_data):
        q0 = imu_data.orientation.w
        q1 = imu_data.orientation.x
        q2 = imu_data.orientation.y
        q3 = imu_data.orientation.z
        yaw = atan2(q1*q2 + q0*q3, 0.5 - q2*q2 - q3*q3)
        self.yaw = yaw

    def img_sub(self, ros_data):
        start = time.time()

        #### direct conversion to CV2 ####
        np_arr = np.fromstring(ros_data.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # save the image for dn
        img_path = '/home/momenta/catkin_ws/src/mc_frenta/mc_detect/nodes/buff.jpg'
        cv2.imwrite(img_path,image)

        odom = self.get_odom()
        self.previous_coordinate['robot'] = np.array((odom[0].x,odom[0].y))-np.array((self.pre_odom[0].x,self.pre_odom[0].y)) \
                                            + np.array(self.previous_coordinate['robot'])
        # self.coordinate = self.previous_coordinate
        self.coordinate = coordinate_calculate(self.location, -self.yaw, self.previous_coordinate)
        self.pre_odom = odom
        self.previous_coordinate = self.coordinate

        if len(self.his_odom) == 0:
            self.his_odom = [odom for _ in range(self._N_FRAME)]
        else:
            del self.his_odom[0]
            self.his_odom.append(odom)

        if len(self.his_coordinate) == 0:
            self.his_coordinate = [self.coordinate for _ in range(self._N_FRAME)]
        else:
            del self.his_coordinate[0]
            self.his_coordinate.append(self.coordinate)
        # publish detector
        result = dn.detect(self.net, self.meta, bytes(img_path))
        self.pub_processed_image(image,result)

        # publish map
        self.map_publisher()


        # read the raw image for processing
        self.location = location_detect(image, result)

        if len(self.his_loca) == 0:
            self.his_loca = [self.location for _ in range(self._N_FRAME)]
        else:
            del self.his_loca[0]
            self.his_loca.append(self.location)

        # 所有帧都没球才标记为没球
        if self.contain_ball() is False:
            self.ready = False
            self.found = False
        else:
            theta_lt_zero = [loc['ball']['theta'] > 0 for loc in self.his_loca]
            if theta_lt_zero.count(True) == 5:
                # 所有帧的 theta 都大于 0
                self.turn_flag = False
            else:
                self.turn_flag = True

        # if self.location['contain_ball'] == 0 :
        #     self.ready = False
        #     self.found = False
        # else:
        #     # self.found = True
        #     if self.location['ball']['theta'] < 0:
        #         self.turn_flag = True
        #     else:
        #         self.turn_flag = False

        # print('yaw',self.yaw*57.29578)
        # print('odom.x:%f,odom.y:%f'%(odom[0].x,odom[0].y))
        # print('location',self.location)
        # print('coordinate',self.coordinate)

        self.start = True
        end = time.time()
        # print('running_time:  \n', end - start)
        
    def map_publisher(self):
        map = np.copy(self.map)

        ball = self.coordinate['ball']
        robot = self.coordinate['robot']
        enemy = self.coordinate['enemy_robot']
        door1 = np.array((2.6, 1.2))
        door2 = np.array((2.6, 0.5))

         # draw line
        if ball[0] > door1[0]/2:
            line1y = ((door1[0]*ball[1] - door1[0]*door1[1])/(door1[0] -ball[0])+door1[1])*200
            line2y = ((door2[0]*ball[1] - door2[0]*door2[1])/(door2[0] -ball[0])+door2[1])*200
            cv2.line(map,(0,line1y.astype(int)),(520,240),(0,0,0),thickness = 1)
            cv2.line(map,(0,line2y.astype(int)),(520,100),(0,0,0),thickness = 1)
        else:
            k = tan(self.ktheta)
            y0 = int(ball[1]*200 - k * ball[0]*200)
            y1 = int(ball[1]*200 + k * ball[0]*200)
            cv2.line(map,tuple((ball*100*2).astype(int)),(520,240),(0,0,0),thickness = 1)
            cv2.line(map,tuple((ball*100*2).astype(int)),(520,100),(0,0,0),thickness = 1)
            cv2.line(map,(0,y0),(int(ball[0]*200),int(ball[1]*200)),(0,0,0),thickness = 1)
            cv2.line(map,(0,y1),(int(ball[0]*200),int(ball[1]*200)),(0,0,0),thickness = 1)

        # draw ball
        ball = tuple((ball*100*2).astype(int))
        cv2.circle(map,ball,14,(255,255,255),thickness=2)
        cv2.circle(map,ball,2,(255,255,255),thickness=2)


        right = self.is_right()
        # draw robot and enemy robot
        robot = robot*100*2
        cv2.rectangle(map,tuple((robot-10).astype(int)),tuple((robot+10).astype(int)),(255,0,0)if right else (255,255,255) ,thickness = 2)
        enemy = enemy*100*2
        cv2.rectangle(map,tuple((enemy-10).astype(int)),tuple((enemy+10).astype(int)),(0,0,255) ,thickness = 2)
        map = np.flipud(map)
        map = np.fliplr(map)

   
        map = map.transpose([1,0,2])
        #### Create CompressedIamge ####
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', map)[1]).tostring()
        # Publish new image
        self.map_pub.publish(msg)

    def pub_processed_image(self,image,result):
        # draw_box
        for i in range(len(result)):
            x = int(result[i][2][0])
            y = int(result[i][2][1])
            w_1_2 = int(result[i][2][2]/2)
            h_1_2 = int(result[i][2][3]/2)
            property = str(result[i][0])
            classes = self.all_classes.index(property)
            box_color = self.box_colors[classes]
            cv2.rectangle(image, (x - w_1_2, y - h_1_2), (x + w_1_2, y + h_1_2), box_color, 2)
            cv2.rectangle(image, (x - w_1_2, y - h_1_2 - 20),
                          (x + w_1_2, y - h_1_2), (125, 125, 125), -1)
            cv2.putText(image, property + ' : %.2f' % result[i][1], (x - w_1_2 + 5, y - h_1_2 - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        #### Create CompressedIamge ####
        msg = CompressedImage()
        msg.header.stamp = rospy.Time.now()
        msg.format = "jpeg"
        msg.data = np.array(cv2.imencode('.jpg', image)[1]).tostring()
        # Publish new image
        self.image_pub.publish(msg)

    def ctrl(self):
        # pass
        # method need to be implemented is noted as TODO

        # TODO:1st step: get to the position in front of the ball fast!  
        self.go_to_point1()
        while not rospy.is_shutdown(): 

        #     # ctrl start only if image detector running
            if self.start: 
                # whether the robot is in the right area to push the ball
                # and is ready to push (self.ready will be false if loss track
                # the ball,during finding and going to the right position)
                # if remove the ready flag , the robot will start pushing the ball
                # when reach the edge of the right area
                if self.is_right() and self.ready:
                    print('push ball')
                    self.push_ball(self.location)
                else:
                    # turn to find the ball
                    # turn flag is last tracking in the ball's position,left or right
                    print('turn to find')
                    self.turn_to_find()
                    # TODO: aim to the ball close to it.
                    print('reach ball')
                    self.reach_ball()
                    # TODO: turn to the right position
                    print('reach position')
                    self.reach_position()

                    # # when get to the right position, it is ready to push the ball
                    self.ready = True

    def go_to_point1(self):
        (position, rotation) = self.get_odom()
        (goal_x, goal_y, goal_z) = (1.3,0.9,0)
        # get to the goal position
        distance= sqrt(pow(goal_x - position.x, 2) + pow(goal_y - position.y, 2))
        r = rospy.Rate(20)
        move_cmd = Twist()
        last_rotation = 0
        linear_speed = 1
        angular_speed = 1.5
        while distance > 0.1 :
            (position, rotation) = self.get_odom()
            x_start = position.x
            y_start = position.y
            path_angle = atan2(goal_y - y_start, goal_x- x_start)

            if path_angle < -pi/4 or path_angle > pi/4:
                if goal_y < 0 and y_start < goal_y:
                    path_angle = -2*pi + path_angle
                elif goal_y >= 0 and y_start > goal_y:
                    path_angle = 2*pi + path_angle
            if last_rotation > pi-0.1 and rotation <= 0:
                rotation = 2*pi + rotation
            elif last_rotation < -pi+0.1 and rotation > 0:
                rotation = -2*pi + rotation
            move_cmd.angular.z = angular_speed * path_angle-rotation

            distance = sqrt(pow((goal_x - x_start), 2) + pow((goal_y - y_start), 2))
            move_cmd.linear.x = min(linear_speed * distance, 0.22)

            if move_cmd.angular.z > 0:
                move_cmd.angular.z = min(move_cmd.angular.z, 2.84)
            else:
                move_cmd.angular.z = max(move_cmd.angular.z, -2.84)

            last_rotation = rotation
            self.cmd_vel.publish(move_cmd)
            r.sleep()
        # (position, rotation) = self.get_odom()

        # while abs(rotation - goal_z) > 0.05:
        #     (position, rotation) = self.get_odom()
        #     if goal_z >= 0:
        #         if rotation <= goal_z and rotation >= goal_z - pi:
        #             move_cmd.linear.x = 0.00
        #             move_cmd.angular.z = 1
        #         else:
        #             move_cmd.linear.x = 0.00
        #             move_cmd.angular.z = -1
        #     else:
        #         if rotation <= goal_z + pi and rotation > goal_z:
        #             move_cmd.linear.x = 0.00
        #             move_cmd.angular.z = -1
        #         else:
        #             move_cmd.linear.x = 0.00
        #             move_cmd.angular.z = 1
        #     self.cmd_vel.publish(move_cmd)
        #     r.sleep()
        # self.cmd_vel.publish(Twist())

    def push_ball(self):
        '''
        aim to the ball and push
        '''
        r = rospy.Rate(10)
        if self.contain_ball():
            theta = self.get_his_theta()
            theta_lt_ten = [abs(t) < 10 for t in theta]
            # if abs(theta) < 10
            if True in theta_lt_ten:
                print('find the ball at center')
                move_cmd = Twist()
                move_cmd.linear.x = 0.22
                self.cmd_vel.publish(move_cmd)
            else:
                print('find the ball, center to it, theta:%f'%theta)
                move_cmd = Twist()
                move_cmd.angular.z = (1 if self.turn_flag else -1)*0.1
                move_cmd.linear.x = 0.22
                time.sleep(0.2)
                self.cmd_vel.publish(move_cmd)
        else:
            move_cmd = Twist()
            move_cmd.linear.x = -0.22
            self.cmd_vel.publish(move_cmd)

        r.sleep()

    def turn_to_find(self):
        while not self.found:
            r = rospy.Rate(10)
            if self.contain_ball() is False:
            # if self.location['contain_ball']==0:
                move_cmd = Twist()
                move_cmd.angular.z = FIND_BALL_ANGULAR_Z*(1 if self.turn_flag else -1)
                self.cmd_vel.publish(move_cmd)
            else:
                # theta = self.location['ball']['theta']
                his_theta = self.get_his_theta()
                theta_le_ten = [t < 10 for t in his_theta]
                if True in theta_le_ten :
                    self.found = True
                else:
                    move_cmd = Twist()
                    move_cmd.angular.z = TURN_BALL_ANGULAR_Z * (1 if self.turn_flag else -1)
                    self.cmd_vel.publish(move_cmd)
            r.sleep()

    def reach_ball(self):
        while (not self.is_right()) and self.found:
            r = rospy.Rate(10)
            theta = self.location['ball']['theta']

            if theta < 0:
                self.turn_flag = True
            else:
                self.turn_flag = False

            if self.get_ball_dist() > 170:
                move_cmd = Twist()
                move_cmd.linear.x = 1
                move_cmd.angular.z = 0.1 if self.turn_flag else -0.1
                self.cmd_vel.publish(move_cmd)

            elif self.get_ball_dist() < 120:
                move_cmd = Twist()
                move_cmd.linear.x = -1
                move_cmd.angular.z = 0.1 if self.turn_flag else -0.1
                self.cmd_vel.publish(move_cmd)
            else:
                break
            r.sleep()

    def contain_ball(self, always=False):
        contain_ball = [loc['contain_ball'] != 0 for loc in self.his_loca]
        # 所有帧都没球才标记为没球
        if contain_ball.count(False) == self._N_FRAME:
            return False
        else:
            if always:
                # 每一帧都检测到了球
                if contain_ball.count(True) == 5:
                    return True
                else:
                    return False
            return True

    def get_coor_ball(self):
        his_ball = [coor['ball'] for coor in self.his_coordinate]
        his_ball.sort()

        ball = np.zeros(2)
        ball[0] = sum(his_ball[1:-1][0]) / (self._N_FRAME - 2)
        ball[1] = sum(his_ball[1:-1][1]) / (self._N_FRAME - 2)

        return ball

    def get_coor_robot(self):
        his_robot = [coor['robot'] for coor in self.his_coordinate]
        his_robot.sort()

        robot = np.zeros(2)
        robot[0] = int(float(sum(his_robot[1:-1][0])) / (self._N_FRAME-2))
        robot[1] = int(float(sum(his_robot[1:-1][1])) / (self._N_FRAME-2))

        return robot

    def get_his_theta(self):
        theta = []
        for i in range(self._N_FRAME):
            if self.his_loca[i]['contain_ball']:
                theta.append(self.his_loca[i]['ball']['theta'])
        return theta

    def get_ball_dist(self):
        dist = []
        for i in  range(self._N_FRAME):
            location = self.his_loca[i]
            if location['contain_ball']:
                dist.append(location['ball']['dist'])
        return int(float(sum(dist)) / len(dist))

    def reach_position(self):
        door = np.array((2.6, 0.85))

        (goal_x, goal_y, goal_z) = self.cal_aim1(self.coordinate)

        # ball = self.coordinate['ball']
        # robot = self.coordinate['robot']

        ball = self.get_coor_ball()
        robot = self.get_coor_robot()

        if robot[0]*(door[1]-ball[1])/(door[0] - ball[0]) + (door[0]*ball[1] - door[0]*door[1])/(door[0] -ball[0])+door[1]-robot[1]>0:
            turn = True
        else:
            turn = False
        rotation = self.yaw
        print('turn 90 ')
        # print(rotation)
        r = rospy.Rate(10)

        while not (-0.2<(self.yaw - (rotation+(pi/2 if turn else -pi/2)))<0.2 or  \
            -0.2+2*pi<(self.yaw - (rotation+(pi/2 if turn else -pi/2)))<0.2+2*pi or\
            -0.2-2*pi<(self.yaw - (rotation+(pi/2 if turn else -pi/2)))<0.2-2*pi):
            # print('rotation:',rotation + (pi/2 if turn else -pi/2))
            # print('yaw',self.yaw)
            print('rotaion_theta', self.yaw - (rotation+(pi/2 if turn else -pi/2)))
            move_cmd = Twist()
            move_cmd.angular.z = 0.6 * (1 if turn else -1)
            self.cmd_vel.publish(move_cmd)
            r.sleep()

        print('tiny right')
        # while (not self.tiny_right()) and not (self.contain_ball() and self.location['ball']['is_compeletely_in_vision']):
        while (not self.tiny_right()) and not (self.contain_ball(always=True)):

            move_cmd = Twist()
            move_cmd.linear.x = 0.22
            move_cmd.angular.z = 1.4*(-1 if turn else 1)
            self.cmd_vel.publish(move_cmd)
            r.sleep()


        print('turn to the ball')
        # turn to the ball
        goal = cal_aim1(self.coordinate)
        goal_x, goal_y, goal_z = goal
        while abs(rotation - goal_z) > 0.05:
                    (position, rotation) = self.get_odom()
                    if goal_z >= 0:
                        if rotation <= goal_z and rotation >= goal_z - pi:
                            move_cmd.linear.x = 0.00
                            move_cmd.angular.z = 0.5
                        else:
                            move_cmd.linear.x = 0.00
                            move_cmd.angular.z = -0.5
                    else:
                        if rotation <= goal_z + pi and rotation > goal_z:
                            move_cmd.linear.x = 0.00
                            move_cmd.angular.z = -0.5
                        else:
                            move_cmd.linear.x = 0.00
                            move_cmd.angular.z = 0.5
                    self.cmd_vel.publish(move_cmd)
                    r.sleep()

        # path_angle = atan2(goal_y - y_start, goal_x- x_start)

    def is_right(self):
        door1 = np.array((2.6, 1.2))
        door2 = np.array((2.6, 0.5))
        #ball = self.coordinate['ball']
        #robot = self.coordinate['robot']

        ball = self.get_coor_ball()
        robot = self.get_coor_robot()

        k = tan(self.ktheta)

        if ball[0] > door1[0]/2:
            if robot[0]*(door1[1]-ball[1])/(door1[0] - ball[0]) + (door1[0]*ball[1] - door1[0]*door1[1])/(door1[0] -ball[0])+door1[1]-robot[1]<0\
            and robot[0]*(door2[1]-ball[1])/(door2[0] - ball[0]) + (door2[0]*ball[1] - door2[0]*door2[1])/(door2[0] -ball[0])+door2[1]-robot[1]>0:
                return True
            else:
                return False
        else:
            if robot[1] > k * (robot[0] - ball[0]) + ball[1] and robot[1] < -k * (robot[0] - ball[0]) + ball[1]:
                return True
            else: 
                return False

    def tiny_right(self):
        door1 = np.array((2.6, 1.2))
        door2 = np.array((2.6, 0.5))
        # ball = self.coordinate['ball']
        # robot = self.coordinate['robot']

        ball = self.get_coor_ball()
        robot = self.get_coor_robot()

        k = tan(self.ktheta_tiny)

        if robot[1] > k * (robot[0] - ball[0]) + ball[1] and robot[1] < -k * (robot[0] - ball[0]) + ball[1]:
            return True
        else: 
            return False

    def cal_aim1(self):
        '''
        calculate the aim1's position 

        '''
        # def transform(odom, pose, coordinate):
        #     '''
        #     transform from aim position to the goal postion (In the odom coordinate system)
        #     '''
        #     (position, rotation)  = odom
        #     odometry = np.array((position.x, position.y))
        #     goal = odometry - coordinate['robot'] + pose
        #     return goal
        ball = self.get_coor_ball()
        door = np.array((2.7, 0.85))
        odom = self.get_odom()
        aim = door - (door - ball) * \
        (np.sqrt((door[0] - ball[0])**2 + (door[1] - ball[1])**2) + 0.1) / \
        np.sqrt((door[0] - ball[0])**2 + (door[1] - ball[1])**2)
        # aim = transform(odom, aim, coordinate)
        orientation = np.arctan2((door[1]-ball[1]), (door[0] - ball[0]))
        goal = np.hstack((aim,orientation))
        return goal

    def get_odom(self):
        '''
        get the odom
        '''
        try:
            (trans, rot) = self.tf_listener.lookupTransform(self.odom_frame, self.base_frame, rospy.Time(0))
            rotation = euler_from_quaternion(rot)

        except (tf.Exception, tf.ConnectivityException, tf.LookupException):
            rospy.loginfo("TF Exception")
            return

        return (Point(*trans), rotation[2])

def main(args):
    '''Initializes and cleanup ros node'''
    ic = mc_detect_ctrl()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down ROS Image detector module")

if __name__ == '__main__':
    main(sys.argv)