import os
import shutil
from pathlib import Path
import random

import numpy as np
import yaml
from tdw.add_ons.collision_manager import CollisionManager
from tdw.add_ons.image_capture import ImageCapture
from tdw.add_ons.object_manager import ObjectManager
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils

from sim_objects import RealObj

STATIC_PROP_READ = [{"$type": "send_static_rigidbodies", # necessary for changing static properties in object manager
                     "frequency": "once"},
                    {"$type": "send_segmentation_colors",
                     "frequency": "once"},
                    {"$type": "send_bounds",
                     "frequency": "once"},
                    {"$type": "send_categories",
                     "frequency": "once"}]

rng = np.random.RandomState()

class Processor:
    def __init__(self, config_path, port=1071):
        with open(r'{}'.format(config_path)) as f:
            args = yaml.full_load(f)
	
        port = args['port']
        self.c = Controller(port=port)

        self.global_t = 0
        self.hole_dam_pos = []
        self.accele_th = 0.015
        self.vec_th = 1.0

        self.output_path = args['output_path'] 
        self.max_frame = args['max_frame']
        self.num_sim = args['num_sim']
        self.rand_sleep_interval = args['rand_sleep_interval']
        self.wind = args['apply_wind']
        self.wind_mag = args['wind_mag']
        self.wind_torq = args['wind_torq']
        self.scale = args['scale']


        self.create_env()
        print("Created Environemnt")

        #self.set_cameras() # Set cameras if we want to have image display
        #print("set cameras")

        self.set_obj_manager()
        print("set object manager")

        self.set_collision_manager()
        print("set collision manager")

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        shutil.copyfile(config_path, os.path.join(self.output_path, config_path.split('/')[1]))

    def set_cameras(self):
        # set up camera capture
        self.cameras = ['a']
        camera_a = ThirdPersonCamera(avatar_id="a", #ground
                                     position={"x": 30, "y": 15, "z": 30},
                                     look_at={"x": 0, "y": -1, "z": 0})

        for camera in self.cameras:
            self.c.add_ons.extend([camera_a])

        #path = EXAMPLE_CONTROLLER_OUTPUT_PATH.joinpath("image_only_video")
        path = Path('controller_output')
        path = path.joinpath('image_only_video')
        print(f"Images at: {path.resolve()}")
        capture = ImageCapture(path=path, avatar_ids=["a"])
        self.c.add_ons.append(capture)

    def create_env(self):
        # defines exterior room & initializes environment
        self.c.communicate([#TDWUtils.create_empty_room(50, 50),
            {"$type": "perlin_noise_terrain",
             "size": {"x": 150, "y": 150},
             "subdivisions": 1,
             "turbulence": 0,
             "origin": {"x": 0, "y": 0},
             "texture_scale": {"x": 10, "y": 10},
             "dynamic_friction": 1.0,
             "static_friction": 1.0,
             "bounciness": 0.8,
             "max_y": 10},
            {"$type": "set_render_quality", "render_quality": 1,}, ])
            #{"$type": "set_screen_size", "width": 1000, "height": 1000}])

    def set_obj_manager(self):
        self.object_manager = ObjectManager(transforms=True, rigidbodies=True, bounds=True)
        self.c.add_ons.append(self.object_manager)

    def set_collision_manager(self):
        self.collision_manager = CollisionManager(enter=True, stay=True, exit=True, objects=True, environment=True)
        self.c.add_ons.append(self.collision_manager)


    def create_frozen_balls(self, sim_id, num_obj=2):
        # creates apples suspended in place
        self.real_water_objs = []
        self.obj_ids = []
        commands = []

        obj_states = [False, True] # false is the noraml state
        position_o1 = [rng.randint(-10*10, 10*10) / 10,
                       rng.randint(4*10, 6.5*10) / 10,
                       rng.randint(-10*10, 10*10) / 10
                       ]
        scale = self.scale
        position_o2 = [position_o1[0] + rng.randint(-scale*10, scale*10) / 100,
                       rng.randint(7.5*10, 9*10) / 10, #round(rng.randint(5*10, 9*10) / 10, 1),
                       position_o1[2] + rng.randint(-scale*10, scale*10) / 100
                       ]

        positions = [position_o1, position_o2]
        for oid in range(num_obj):
            obj_id = self.c.get_unique_id()

            position = positions[oid]


            obj = RealObj()
            obj.position.append([self.global_t, position[0], position[1], position[2]])
            obj.rotation.append([self.global_t, 0, 0, 0])
            obj.forward.append([self.global_t, 0, 0, 0])
            obj.angular_velocity.append([self.global_t, 0, 0, 0])
            obj.velocity.append([self.global_t, 0, 0, 0])
            obj.obj_id = obj_id
            obj.meta_data['start_t'] = self.global_t
            obj.meta_data['obj_id'] = obj_id
            obj.meta_data['sim_id'] = sim_id
            obj.meta_data['start_pos'] = tuple(position)

            commands.extend(self.c.get_add_physics_object(model_name="prim_sphere",
                                                          object_id=obj_id,
                                                          library='models_special.json',
                                                          scale_factor={"x": 1, "y": 1, "z": 1},
                                                          position={"x": position[0], # -6, 4
                                                                    "y": position[1],
                                                                    "z": position[2]}, # 6, -4
                                                          bounciness=0.8,
                                                          dynamic_friction=1.0,
                                                          static_friction=1.0,
                                                          mass=0.1,
                                                          gravity=True,
                                                          kinematic=obj_states[oid]))
            self.real_water_objs.append(obj)
            self.obj_ids.append(obj_id)
            # print('apples, ', obj_id)
        self.c.communicate(commands=commands)
        self.set_collision_manager()
        self.global_t += 1

    def apply_wind(self, obj):
        windx = self.wind_mag[0] #float(rng.uniform(-10, 10))  # can be more than 1
        windy = self.wind_mag[1] #float(rng.uniform(-5, 5)) 
        windz = self.wind_mag[2] #float(rng.uniform(-10, 10))

        torquex = self.wind_torq[0]
        torquey = self.wind_torq[1]
        torquez = self.wind_torq[2]
        command = [{"$type": "apply_force_to_object", "id": obj.obj_id, "force": {"x": windx, "y": windy, "z": windz}},
                   {"$type": "apply_torque_to_object", "id": obj.obj_id, "force":
                       {"x": torquex, "y": torquey, "z": torquez}}]

        return command, (windx, windy, windz, torquex, torquey, torquez)


    def run_single_sim(self, sim_id, applyWind=False):
        done = False
        updated_obj = set()

        t_to_apply_2nd_ball = rng.randint(10, 80)
        if applyWind:
            rand_wind_flag = True #if rng.random(1) >= 0.5 else False 

            obj_to_be_applied = rng.randint(0, 2)
            if obj_to_be_applied == 1:
                t_to_apply = t_to_apply_2nd_ball + rng.randint(1, 21) #rng.randint(t_to_apply_2nd_ball, 140)
            else:
                t_to_apply = rng.randint(20, 70) #rng.randint(10, 140)
            # stop_apply_wind = [rng.randint(7, 8) for _ in range(len(self.real_water_objs))]

        single_t = 0
        while not done and single_t < self.max_frame: # make sure all waters will contact the ground.
            # read every obj's states
            # ('global time.... ', self.global_t)
            commands = []

            tmp_done = [True, True]

            for idx, obj in enumerate(self.real_water_objs):
                pos = self.object_manager.transforms[obj.obj_id].position
                rot = self.object_manager.transforms[obj.obj_id].rotation
                forw = self.object_manager.transforms[obj.obj_id].forward
                vec = self.object_manager.rigidbodies[obj.obj_id].velocity
                ang = self.object_manager.rigidbodies[obj.obj_id].angular_velocity
                sleeping = self.object_manager.rigidbodies[obj.obj_id].sleeping
                # if idx == 1:
                #     print(single_t, pos)

                # early stop condition check
                early_stop = False

                for col_obj_id in self.collision_manager.obj_collisions: # obj collision
                #    # print(col_obj_id, obj.obj_id, self.collision_manager.obj_collisions[col_obj_id].state)
                    if obj.obj_id in [col_obj_id.int1, col_obj_id.int2] and \
                            self.collision_manager.obj_collisions[col_obj_id].state=='enter':
                        # obj.isCollide = True
                        obj.meta_data['obj_collision_num'] += 1
                        print('OBJ collision')

                for col_obj_id in self.collision_manager.env_collisions:  # 0env collision
                    if col_obj_id == obj.obj_id and \
                            self.collision_manager.env_collisions[obj.obj_id].state=='enter':
                        obj.isCollide = True
                        obj.meta_data['collision_num'] += 1
                        print('ENV collision', single_t)

                # check a ball's acceleration
                if obj.isCollide and len(obj.velocity) >= 100 and single_t > t_to_apply_2nd_ball:
                    accele = []
                    vec_arr = []
                    vec_arr_y = []
                    vec_arr_y2 = []
                    for vec_i in range(-1, -6, -1):
                        curr_x, curr_y, curr_z = obj.velocity[vec_i][1:]
                        last_x, last_y, last_z = obj.velocity[vec_i-1][1:]

                        curr_v = ((curr_x)**2+(curr_y)**2+(curr_z)**2) ** 0.5
                        last_v = ((last_x)**2+(last_y)**2+(last_z)**2) ** 0.5
			 
                        vec_arr_y.append(abs(curr_y))
                        vec_arr_y2.append(abs(self.real_water_objs[1].velocity[vec_i][2]))
			 
                        accele.append(abs(curr_v - last_v))
                        vec_arr.append(curr_v)
                    ave_accele = sum(accele) / len(accele)
                    ave_vec = sum(vec_arr) / len(vec_arr)
                    ave_vec_y = sum(vec_arr_y) / len(vec_arr_y)
                    ave_vec_y2 = sum(vec_arr_y2) / len(vec_arr_y2)
                    #if idx == 0:
                    #print('ave, ', single_t, ave_vec, ave_accele, ave_vec_y)
                    if ave_accele <= self.accele_th and ave_vec <= self.vec_th and ave_vec_y == 0 and ave_vec_y2 == 0:
                        early_stop = True
                        print('acceleration or velocity below the sleep threshold')


                if obj.obj_id not in updated_obj:
                    # pos = np.round(pos, 2)
                    obj.position.append([self.global_t] + list(pos))
                    obj.rotation.append([self.global_t] + list(rot))
                    obj.forward.append([self.global_t] + list(forw))
                    obj.velocity.append([self.global_t] + list(vec))
                    obj.angular_velocity.append([self.global_t] + list(ang))

                tmp_done[idx] = tmp_done[idx] & (sleeping | early_stop)

                if (sleeping or early_stop) and single_t > t_to_apply_2nd_ball: # Update land
                    obj.meta_data['end_t'] = obj.position[-1][0]
                    obj.meta_data['duration'] = self.global_t - obj.meta_data['start_t']
                    updated_obj.add(obj.obj_id)
                    # print("stopped: ", single_t, sleeping, early_stop)

                if applyWind and not (sleeping or early_stop) and rand_wind_flag:
                    if idx == obj_to_be_applied and single_t == t_to_apply:
                        wind_cmd, wind_mag = self.apply_wind(obj)
                        commands.extend(wind_cmd)
                        obj.meta_data['wind_magnitude'] = tuple([self.global_t]) + wind_mag
                        obj.meta_data['apply_wind'] = True
                        print('applying wind: ', wind_mag)

            if single_t == t_to_apply_2nd_ball:
                self.real_water_objs[1].meta_data['start_t'] = self.global_t
                commands.extend([{"$type": "set_kinematic_state",
                                  "id": self.real_water_objs[1].obj_id,
                                  "is_kinematic": False, "use_gravity": True}])

            commands.extend(STATIC_PROP_READ) # necessary for changing static properties in object manager
            resp = self.c.communicate(commands) # advance
            self.object_manager._cached_static_data = False # goto the object_manager source code to see why.
            self.object_manager.on_send(resp)
            self.global_t += 1
            single_t += 1

            done = tmp_done[0]


        # when outside the loop, check the non-updated water position
        for idx, obj in enumerate(self.real_water_objs):
            if obj.obj_id not in updated_obj:
                obj.meta_data['end_t'] = obj.position[-1][0]
                obj.meta_data['duration'] = self.global_t - obj.meta_data['start_t']

            obj.save_to_file(self.output_path, sim_id, idx)
            self.c.communicate({"$type": "destroy_object",
                                "id": obj.obj_id})

        # Mark the object manager as requiring re-initialization.
        self.object_manager.initialized = False
        self.collision_manager.initialized = False

    # complete simulation with randomness
    def process(self):
        for sim_id in range(self.num_sim):
            print("sim {}.......................".format(sim_id))
            self.create_frozen_balls(sim_id)
            self.run_single_sim(sim_id, self.wind)
            print('single run finish, sleeping')
            rng_interval = rng.randint(1, self.rand_sleep_interval)
            commands = []
            for t in range(self.global_t, self.global_t+rng_interval):

                commands.extend(STATIC_PROP_READ) # necessary for changing static properties in object manager

                resp = self.c.communicate(commands) # advance
                self.object_manager._cached_static_data = False # goto the object_manager source code to see why.
                self.object_manager.on_send(resp)

            self.global_t += rng_interval


        self.c.communicate({"$type": "terminate"})
        print("sim end")

if __name__ == '__main__':
    proc = Processor('land_params_V3.1.yaml')
    proc.process()
