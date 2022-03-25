import OpenGL

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import glfw

import pygame
from pygame.locals import *

import pywavefront

import numpy as np
import tempfile
import pathlib
import json
import dacite
from scipy.integrate import odeint

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

import contracts

def parse(self):
    try:
        while True:
            if not self.line:
                self.next_line()

            if self.line[0] == '#' or self.line[0] == 'g' or len(self.values) < 2:
                self.consume_line()
                continue
            self.dispatcher.get(self.values[0], self.parse_fallback)()
    except StopIteration:
        pass

    if self.auto_post_parse:
        self.post_parse()
        
pywavefront.parser.Parser.parse = parse

class HotRocket:
        
    def load_model(self, path):
        self.model_path = pathlib.Path(path)
        
        file_contents = open(self.model_path, "r").readlines()
        
        v_lines = [line for line in file_contents if line.startswith('v')]        
        f_lines = []
        for line in file_contents:
            if line.startswith("g"):
                f_lines.append([])
            if line.startswith("f"):
                f_lines[-1].append(line)
            
        self.parts = []
        for f_line_group in f_lines:
            
            tmp = tempfile.TemporaryFile('w', delete=False, suffix=".obj")
            tmp.write("".join(v_lines))
            tmp.write("".join(f_line_group))
            tmp.close()   
            self.parts.append(self.init_scene(tmp.name))
            
            os.remove(tmp.name)
    
    def load_info(self, info_path):
        self.info_path = pathlib.Path(info_path)
        info = json.load(open(self.info_path, encoding='utf-8'))
        for p, i in zip(self.parts, info):
            p.info = dacite.from_dict(data_class=contracts.PartInfo, data=i, config=dacite.Config(type_hooks=contracts.converters))
        
        self.parts = list(sorted(self.parts, key=lambda p: p.info.id))
        self.init_calc_data()
        
    def save_solution(self, path):
        np.savetxt(
            path, 
            self.sol, 
            header=",".join(str(i) for i in range(self.sol.shape[1])),
            delimiter=",",
            comments='',
            fmt="%.4e"
        )
        
    def Q_TC(self, y, t):
        return (y[:, None] - y) * self.k
       
    def Q_E(self, y):
        return -self.eps * self.squares * self.C_0 * (y / 100) ** 4
    
    def Q_R(self, t):
        result = np.zeros(len(self.parts))
        for k in range(len(self.parts)):
            result[k] = self.parts[k].info.Q_R.func(self, t)
        return result
        
    def calc_deriv(self, y, t):
        return (np.sum(self.Q_TC(y, t), axis=0) + self.Q_E(y) + self.Q_R(t)) / self.c
        
    def init_calc_data(self):
        self.k = np.zeros((len(self.parts), len(self.parts)))
        
        self.c = np.zeros((len(self.parts)))
        self.eps = np.zeros((len(self.parts),))
        self.squares = np.zeros((len(self.parts,)))
        self.C_0 = 5.67
        
        for part in sorted(self.parts, key=lambda p: p.info.id):
            for info in part.info.connection_info:
                self.k[part.info.id, info.id] = info.lmbda * info.square
                self.eps[part.info.id] = part.info.eps
                self.squares[part.info.id] = part.info.square
                self.c[part.info.id] = part.info.c
        
    def _set_T(self, T):
        self.T = T

    def _set_num_points(self, num_points):
        self.num_points = num_points
        
    def solve_ode(self):
        y0 = np.array([25, 20, 20, 20, 20])
        self.t = np.linspace(0, self.T, self.num_points)
        self.sol = odeint(self.calc_deriv, y0, self.t)
        self.maximum = np.max(self.sol)
        self.minimum = np.min(self.sol)
        self.cur_t = 0      
        
    def init_scene(self, path):
        
        scene = pywavefront.Wavefront(path, collect_faces=True)

        scene_box = (scene.vertices[0], scene.vertices[0])
        for vertex in scene.vertices:
            min_v = [min(scene_box[0][i], vertex[i]) for i in range(3)]
            max_v = [max(scene_box[1][i], vertex[i]) for i in range(3)]
            scene_box = (min_v, max_v)

        scene_size     = [scene_box[1][i]-scene_box[0][i] for i in range(3)]
        max_scene_size = max(scene_size)
        scaled_size    = 5
        scene_scale    = [scaled_size/max_scene_size for i in range(3)]
        scene_trans    = [-(scene_box[1][i]+scene_box[0][i])/2 for i in range(3)]
        
        return contracts.Part(scene, scene_scale, scene_trans)
    
    def init_gl(self):
        if not glfw.init():
            return
        glfw.window_hint(glfw.VISIBLE, False)
        self.window = glfw.create_window(self.width, self.height, "hidden window", None, None)
        if not self.window:
            glfw.terminate()
            return

        glfw.make_context_current(self.window)
        gluPerspective(45, (self.width / self.height), 1, 500.0)
        glTranslatef(0.0, 0.0, -10)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        
    def update_curve_plot(self):
        self.curve_ax.clear()
        self.curve_plot = self.curve_ax.plot(self.t, self.sol)
        
    def init_axis(self):
        self.fig = plt.figure(figsize=(self.width/80,self.height/80), dpi=80)
        
        self.gl_ax = self.fig.add_subplot(122)
        self.gl_ax.axis('off')

        self.curve_ax = self.fig.add_subplot(121)
        self.update_curve_plot()

        cmap = cm.get_cmap("plasma")
        norm = matplotlib.colors.Normalize(vmin=self.minimum, vmax=self.maximum)
        
        self.img_plot = self.gl_ax.imshow(
            np.zeros((self.width, self.height)), aspect='auto',
        )
        plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=self.gl_ax)
    
    def __init__(self, model_path, width=300, height=300, info_path=None):
        
        self.width = width
        self.height =  height
        
        self.load_model(model_path)
        if info_path:
            self.load_info(info_path)
        
        self.init_gl()
        
        self._set_T(1000)
        self._set_num_points(1000)
        
        self.solve_ode()
        self.init_axis()
                
    def process_event(self, event):
        pass
        
    def update(self, time_delta):
        pass

    def map_t_to_color(self, idx):
        return cm.get_cmap("plasma")(
            (self.sol[self.cur_t, idx] - self.minimum)/(self.maximum - self.minimum)
        )

    def image_to_plot(self, image):
        self.img_plot.set(data=image)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        image = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype='uint8')
        return image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

    def draw(self, screen):

        glRotatef(1, 5, 5, 0)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        
        glPushMatrix()
        glScalef(*self.parts[0].scene_scale)
        glTranslatef(*self.parts[0].scene_trans)
        
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        
        for k, part in enumerate(self.parts):
            for mesh in part.scene.mesh_list:
                current_color = self.map_t_to_color(k)
                glColor3f(*current_color[:3])
                # glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, current_color)
                self.cur_t = (self.cur_t + 1) % self.sol.shape[0]
                glBegin(GL_TRIANGLES)
                for face in mesh.faces:
                    for vertex_i in face:
                        glVertex3f(*part.scene.vertices[vertex_i])
                glEnd()

        glPopMatrix()
        
        image_buffer = glReadPixels(0, 0, self.width, self.height, OpenGL.GL.GL_RGB, OpenGL.GL.GL_UNSIGNED_BYTE)
        image = np.frombuffer(image_buffer, dtype=np.uint8).reshape(self.height, self.width, 3)
        image = self.image_to_plot(image)
        
        tmp_surf = pygame.pixelcopy.make_surface(image.swapaxes(0, 1))
        screen.blit(tmp_surf, (0, 0))
    
    def __del__(self):
        glfw.destroy_window(self.window)
        glfw.terminate()