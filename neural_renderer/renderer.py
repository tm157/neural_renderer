import math

import chainer.functions as cf

import neural_renderer
import ipdb


class Renderer(object):
    def __init__(self):
        # rendering
        self.image_size = 256
        self.anti_aliasing = True
        self.background_color = [0, 0, 0]
        self.fill_back = True

        # camera
        self.perspective = True
        self.viewing_angle = 30
        self.eye = [0, 0, -(1. / math.tan(math.radians(self.viewing_angle)) + 1)]
        self.camera_mode = 'look_at'
        self.camera_direction = [0, 0, 1]
        self.near = 0.1
        self.far = 100

        # light
        self.light_intensity_ambient = 0.5
        self.light_intensity_directional = 0.5
        self.light_color_ambient = [1, 1, 1]  # white
        self.light_color_directional = [1, 1, 1]  # white
        self.light_direction = [0, 1, 0]  # up-to-down

        # rasterization
        self.rasterizer_eps = 1e-3

    def render_silhouettes(self, vertices, faces):
        # fill back
        if self.fill_back:
            faces = cf.concat((faces, faces[:, :, ::-1]), axis=1).data

        # viewpoint transformation
        if self.camera_mode == 'look_at':
            vertices = neural_renderer.look_at(vertices, self.eye)
        elif self.camera_mode == 'look':
            vertices = neural_renderer.look(vertices, self.eye, self.camera_direction)

        # perspective transformation
        if self.perspective:
            vertices = neural_renderer.perspective(vertices, angle=self.viewing_angle)

        # rasterization
        faces = neural_renderer.vertices_to_faces(vertices, faces)
        # ==== TM changes ====
        results_dict = neural_renderer.rasterize_silhouettes(
                faces, self.image_size, self.anti_aliasing)
        masks = results_dict['alpha']
        face_index_map = results_dict['face_index_map']
        weight_map = results_dict['weight_map']
        sampling_weight_map = results_dict['sampling_weight_map']
        # ==== Making another dictionary (just for clarity) ====
        return_dict = dict()
        return_dict['masks'] = masks
        return_dict['face_index_map'] = face_index_map
        return_dict['weight_map'] = weight_map
        return_dict['sampling_weight_map'] = sampling_weight_map
        return return_dict
        # ==== End ====

    def render_depth(self, vertices, faces):
        # fill back
        if self.fill_back:
            faces = cf.concat((faces, faces[:, :, ::-1]), axis=1).data

        # viewpoint transformation
        if self.camera_mode == 'look_at':
            vertices = neural_renderer.look_at(vertices, self.eye)
        elif self.camera_mode == 'look':
            vertices = neural_renderer.look(vertices, self.eye, self.camera_direction)

        # perspective transformation
        if self.perspective:
            vertices = neural_renderer.perspective(vertices, angle=self.viewing_angle)

        # rasterization
        faces = neural_renderer.vertices_to_faces(vertices, faces)
        images = neural_renderer.rasterize_depth(faces, self.image_size, self.anti_aliasing)
        return images

    def render(self, vertices, faces, textures):
        # fill back
        if self.fill_back:
            faces = cf.concat((faces, faces[:, :, ::-1]), axis=1).data
            textures = cf.concat((textures, textures.transpose((0, 1, 4, 3, 2, 5))), axis=1)

        # lighting
        faces_lighting = neural_renderer.vertices_to_faces(vertices, faces)
        textures = neural_renderer.lighting(
            faces_lighting,
            textures,
            self.light_intensity_ambient,
            self.light_intensity_directional,
            self.light_color_ambient,
            self.light_color_directional,
            self.light_direction)

        # viewpoint transformation
        if self.camera_mode == 'look_at':
            vertices = neural_renderer.look_at(vertices, self.eye)
        elif self.camera_mode == 'look':
            vertices = neural_renderer.look(vertices, self.eye, self.camera_direction)

        # perspective transformation
        if self.perspective:
            vertices = neural_renderer.perspective(vertices, angle=self.viewing_angle)

        # rasterization
        faces = neural_renderer.vertices_to_faces(vertices, faces)
        # ==== TM changes ====
        results_dict = neural_renderer.rasterize(
                faces, textures, self.image_size, self.anti_aliasing, self.near,
                self.far, self.rasterizer_eps, self.background_color)
        images = results_dict['rgb']
        face_index_map = results_dict['face_index_map']
        weight_map = results_dict['weight_map']
        sampling_weight_map = results_dict['sampling_weight_map']

        # ==== Making another dictionary (just for clarity) ====
        return_dict = dict()
        return_dict['images'] = images
        return_dict['face_index_map'] = face_index_map
        return_dict['weight_map'] = weight_map
        return_dict['sampling_weight_map'] = sampling_weight_map
        return return_dict
        # ==== END ====


