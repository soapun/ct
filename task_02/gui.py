import pygame
import pygame_gui

import pygame_gui.elements
from pygame_gui.ui_manager import UIManager
from pygame_gui.windows.ui_file_dialog import UIFileDialog
from rocket import HotRocket

import logging

logging.basicConfig(
     filename='log.log',
     level=logging.DEBUG,
     format= '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
     datefmt='%H:%M:%S'
 )
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)
logger = logging.getLogger(__name__)

class SaveFileDialog(UIFileDialog):
    
    def _validate_file_path(self, path_to_validate):
        return True
    
    def _validate_path_exists_and_of_allowed_type(self, path_to_validate, allow_directories: bool):
        return True

class RocketWindow(pygame_gui.elements.ui_window.UIWindow):
    def __init__(self, model_path, position, ui_manager, info_path=None):
        window_size = (800, 400)
        
        super().__init__(pygame.Rect(position, window_size), ui_manager,
                         window_display_title='Very Hot Rocket',
                         object_id='#rocket_window')

        game_surface_size = self.get_container().get_size()
        self.game_surface_element = pygame_gui.elements.ui_image.UIImage(pygame.Rect((0, 0),
                                                        game_surface_size),
                                            pygame.Surface(game_surface_size).convert(),
                                            manager=ui_manager,
                                            container=self,
                                            parent_element=self)

        self.rocket = HotRocket(model_path, *game_surface_size, info_path)
        self.is_active = False

    def process_event(self, event):
        handled = super().process_event(event)
        if self.is_active:
            handled = self.rocket.process_event(event)
        return handled

    def update(self, time_delta):
        if self.alive() and self.is_active:
            self.rocket.update(time_delta)
        super().update(time_delta)
        self.rocket.draw(self.game_surface_element.image)


class App:
    def __init__(self, size=(1130, 520)):
        pygame.init()

        self.root_window_surface = pygame.display.set_mode(size)
        self.background_surface = pygame.Surface(size).convert()
        self.background_surface.fill(pygame.Color('#505050'))
        self.ui_manager = UIManager(size, 'data/themes/theme_3.json')
        self.clock = pygame.time.Clock()
        self.is_running = True

        default_anchoring = {
            'left': 'left',
            'right': 'right',
            'top': 'top',
            'bottom': 'bottom'
        }
        
        self.save_dialog = None
        self.load_dialog = None
        
        self.rocket_window = RocketWindow(
            r"models\model1.obj", 
            (0,0), 
            self.ui_manager,
            r"models\model1_meta.json",
        )
        
        self.choose_path_button = pygame_gui.elements.ui_button.UIButton(
            pygame.Rect((10, 40), (150, 50)),
            "Load info",
            self.ui_manager,
            anchors={
                'top': 'bottom',
                'left': 'left',
                'bottom': 'bottom',
                'right': 'left',
                'left_target': self.rocket_window,
                'right_target' : self.rocket_window,
                'top_target' : self.rocket_window,
                'bottom_target' : self.rocket_window    
            }
        )
        
        self.save_solution_button = pygame_gui.elements.ui_button.UIButton(
            pygame.Rect((0, 0), (150, 50)),
            "Save solution",
            self.ui_manager,
            anchors={
                'top': 'bottom',
                'left': 'left',
                'bottom': 'bottom',
                'right': 'left',
                'left_target': self.choose_path_button,
                'right_target' : self.choose_path_button,
                'top_target' : self.choose_path_button,
                'bottom_target' : self.choose_path_button    
            }
        )
        
        self.time_label = pygame_gui.elements.ui_label.UILabel(
            pygame.Rect((0, 20), (150, 50)),
            "Calculation time :",
            self.ui_manager,
            anchors={
                'top': 'top',
                'left': 'right',
                'bottom': 'top',
                'right': 'right',
                'left_target': self.choose_path_button,
                'right_target' : self.choose_path_button,
                'top_target' : self.choose_path_button,
                'bottom_target' : self.choose_path_button    
            }
        )
        self.time_entry_line = pygame_gui.elements.ui_text_entry_line.UITextEntryLine(
            pygame.Rect((0, 0), (150, 50)),
            self.ui_manager,
            anchors={
                'top': 'bottom',
                'left': 'left',
                'bottom': 'bottom',
                'right': 'left',
                'left_target': self.time_label,
                'right_target' : self.time_label,
                'top_target' : self.time_label,
                'bottom_target' : self.time_label    
            }
        )
        
        self.num_points_label = pygame_gui.elements.ui_label.UILabel(
            pygame.Rect((0, 20), (150, 50)),
            "Number of points :",
            self.ui_manager,
            anchors={
                'top': 'top',
                'left': 'right',
                'bottom': 'top',
                'right': 'right',
                'left_target': self.time_label,
                'right_target' : self.time_label,
                'top_target' : self.time_label,
                'bottom_target' : self.time_label    
            }
        )
        self.num_points_entry_line = pygame_gui.elements.ui_text_entry_line.UITextEntryLine(
            pygame.Rect((0, 0), (150, 50)),
            self.ui_manager,
            anchors={
                'top': 'bottom',
                'left': 'left',
                'bottom': 'bottom',
                'right': 'left',
                'left_target': self.num_points_label,
                'right_target' : self.num_points_label,
                'top_target' : self.num_points_label,
                'bottom_target' : self.num_points_label    
            }
        )
        

    def _on_choose_path_button(self, event):
        logger.debug("Choose Path Button pressed")
        
        self.load_dialog = UIFileDialog(
            pygame.Rect((200, 200), (400, 400)),
            self.ui_manager,
            visible=False,
        )
        self.load_dialog.enable()
        self.load_dialog.show()
        
    def _on_save_solution_button(self, event):
        logger.debug("Save Solution Button pressed")
        
        self.save_dialog = SaveFileDialog(
            pygame.Rect((200, 200), (400, 400)),
            self.ui_manager,
            visible=False,
            allow_existing_files_only=False,
            initial_file_path="sample.csv"
        )
        self.save_dialog.enable()
        self.save_dialog.show()

    def _on_load_dialog_path_picked(self, event):
        self.rocket_window.rocket.load_info(event.text) 
        
    def _on_save_dialog_path_picked(self, event):
        self.rocket_window.rocket.save_solution(event.text) 

    def _on_time_entry_line_finished(self, event):
        self.rocket_window.rocket._set_T(float(event.text))
        self.rocket_window.rocket.solve_ode()
        self.rocket_window.rocket.update_curve_plot()
        
    def _on_num_points_entry_line_finished(self, event):
        self.rocket_window.rocket._set_num_points(int(event.text))
        self.rocket_window.rocket.solve_ode()
        self.rocket_window.rocket.update_curve_plot()

    def _process_event(self, event):
        if event.type == pygame.QUIT:
            self.is_running = False
            
        self.ui_manager.process_events(event)
        if event.type == pygame.USEREVENT:
            if event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == self.choose_path_button:
                    self._on_choose_path_button(event)
                if event.ui_element == self.save_solution_button:
                    self._on_save_solution_button(event)
                    
            if event.user_type == pygame_gui.UI_FILE_DIALOG_PATH_PICKED:
                if event.ui_element == self.load_dialog:
                    self._on_load_dialog_path_picked(event)
                if event.ui_element == self.save_dialog:
                    self._on_save_dialog_path_picked(event)
                    
            if event.user_type == pygame_gui.UI_TEXT_ENTRY_FINISHED:
                if event.ui_element == self.time_entry_line:
                    self._on_time_entry_line_finished(event)
                if event.ui_element == self.num_points_entry_line:
                    self._on_num_points_entry_line_finished(event)

    def run(self):
        while self.is_running:
            time_delta = self.clock.tick(60)/1000.0

            for event in pygame.event.get():
                self._process_event(event)

            self.ui_manager.update(time_delta)
            self.root_window_surface.blit(self.background_surface, (0, 0))
            self.ui_manager.draw_ui(self.root_window_surface)
            pygame.display.update()