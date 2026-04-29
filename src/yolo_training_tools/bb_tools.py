import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.widgets import Button
import typing
from pathlib import Path

class BoundingBoxPicker:
    """Interactive picker for single/multiple rotated rectangles on an image."""

    def __init__(self, image: typing.Union[str, Path, np.ndarray], normalize: bool = False, rotation_enabled: bool = True, multiple: bool = False):
        self.normalize = normalize
        self.rotation_enabled = rotation_enabled
        self.multiple = multiple

        if isinstance(image, (str, Path)):
            self.img = plt.imread(str(image))
        else:
            self.img = image

        self.img_h, self.img_w = self.img.shape[0], self.img.shape[1]

        self.fig, self.ax = plt.subplots(figsize=(20, 20))
        self.ax.imshow(self.img, origin='upper')
        self.ax.set_xlim(0, self.img_w)
        self.ax.set_ylim(self.img_h, 0)
        self.ax.set_aspect('equal')

        # State variables
        self.angle = 0.0
        self.mode = 'init'
        self.p0 = None
        self.width = 0.0
        self.height = 0.0
        self.u_dir = np.array([1.0, 0.0])
        self.v_dir = np.array([0.0, 1.0])
        self.completed_annotations = []
        self.completed_polygons = []
        self.dragging = None
        self.finished = False

        # Build UI elements
        self._build_ui()
        self._connect_events()
        self.reset_current_annotation()

    def _build_ui(self):
        self.cross1 = Line2D([], [], lw=1, color='cyan')
        self.cross2 = Line2D([], [], lw=1, color='red')
        self.ax.add_line(self.cross1)
        self.ax.add_line(self.cross2)

        self.rect = Polygon([[0, 0]], closed=True, fill=False, lw=2, color='yellow')
        self.ax.add_patch(self.rect)
        self.handles = self.ax.scatter([], [], s=100, color='red', picker=5)

        # Buttons
        self.ax_button_finish = plt.axes([0.85, 0.92, 0.1, 0.05])
        self.btn_finish = Button(self.ax_button_finish, 'Fertig')

        if self.multiple:
            self.ax_button_add = plt.axes([0.85, 0.86, 0.1, 0.05])
            self.btn_add = Button(self.ax_button_add, 'Weitere Box')

            self.ax_button_undo = plt.axes([0.85, 0.80, 0.1, 0.05])
            self.btn_undo = Button(self.ax_button_undo, 'Rückgängig')

    def _connect_events(self):
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)
        if self.rotation_enabled:
            self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        
        self.btn_finish.on_clicked(self.on_finish)
        if self.multiple:
            self.btn_add.on_clicked(self.on_add_another)
            self.btn_undo.on_clicked(self.on_undo)

    def _get_corners(self) -> typing.List[typing.Tuple[float, float]]:
        """Returns ordered corners based on proper width/height dimensions"""
        if self.width >= 0 and self.height >= 0:
            c0, c1 = self.p0, self.p0 + self.u_dir * self.width
            c3, c2 = self.p0 + self.v_dir * self.height, c1 + self.v_dir * self.height
        elif self.width < 0 and self.height >= 0:
            c0, c1 = self.p0 + self.u_dir * self.width, self.p0
            c3, c2 = c0 + self.v_dir * self.height, c1 + self.v_dir * self.height
        elif self.width >= 0 and self.height < 0:
            c0, c1 = self.p0 + self.v_dir * self.height, self.p0 + self.v_dir * self.height + self.u_dir * self.width
            c3, c2 = self.p0, self.p0 + self.u_dir * self.width
        else:
            c0, c1 = self.p0 + self.u_dir * self.width + self.v_dir * self.height, self.p0 + self.v_dir * self.height
            c3, c2 = self.p0 + self.u_dir * self.width, self.p0
            
        return [c0, c1, c2, c3]

    def update_cross(self, x: float, y: float):
        L = max(self.img_w, self.img_h)
        o = np.array([x, y])
        theta = np.deg2rad(self.angle)
        u = np.array([np.cos(theta), np.sin(theta)])
        v = np.array([-np.sin(theta), np.cos(theta)])
        pu = np.vstack([o + u * -L, o + u * L])
        pv = np.vstack([o + v * -L, o + v * L])
        self.cross1.set_data(pu.T)
        self.cross2.set_data(pv.T)
        self.fig.canvas.draw_idle()

    def update_rectangle(self):
        if self.p0 is None: return
        corners = self._get_corners()
        self.rect.set_xy(corners)
        self.handles.set_offsets([corners[1], corners[3]])
        self.fig.canvas.draw_idle()

    def save_current_annotation(self):
        if self.mode == 'edit' and self.p0 is not None:
            corners = self._get_corners()
            corners_tup = [tuple(c) for c in corners]
            abs_w, abs_h = abs(self.width), abs(self.height)
            
            self.completed_annotations.append((corners_tup, abs_w, abs_h, self.angle))
            
            display_corners = corners_tup
            if self.normalize:
                # normalize then un-normalize for display overlay
                display_corners = [(x / self.img_w * self.img_w, y / self.img_h * self.img_h) for x, y in corners_tup]
            
            poly = Polygon(display_corners, closed=True, fill=False, lw=2, color='lime', alpha=0.7)
            self.ax.add_patch(poly)
            self.completed_polygons.append(poly)
            
            self.reset_current_annotation()
            self.fig.canvas.draw_idle()

    def reset_current_annotation(self):
        self.mode = 'init'
        self.p0 = None
        self.angle = 0.0
        self.width = 0.0
        self.height = 0.0
        self.u_dir[:] = [1.0, 0.0]
        self.v_dir[:] = [0.0, 1.0]
        
        self.cross1.set_visible(False)
        self.cross2.set_visible(False)
        self.rect.set_visible(False)
        self.handles.set_visible(False)

    def on_move(self, event):
        if event.inaxes != self.ax or event.xdata is None: return
        x, y = event.xdata, event.ydata
        
        if self.mode == 'init':
            self.cross1.set_visible(True)
            self.cross2.set_visible(True)
            self.update_cross(x, y)
        elif self.mode == 'drawing':
            self.rect.set_visible(True)
            d = np.array([x, y]) - self.p0
            self.width = np.dot(d, self.u_dir)
            self.height = np.dot(d, self.v_dir)
            self.update_rectangle()
        elif self.mode == 'edit' and self.dragging:
            self.rect.set_visible(True)
            self.handles.set_visible(True)
            d = np.array([x, y]) - self.p0
            
            if self.dragging == 'corner1':
                length = np.hypot(*d)
                if length > 1e-3:
                    self.angle = np.rad2deg(np.arctan2(d[1], d[0]))
                    self.width = length
                    theta = np.deg2rad(self.angle)
                    self.u_dir[:] = [np.cos(theta), np.sin(theta)]
                    self.v_dir[:] = [-np.sin(theta), np.cos(theta)]
            elif self.dragging == 'corner2':
                length = np.hypot(*d)
                if length > 1e-3:
                    vx, vy = d / length
                    self.angle = np.rad2deg(np.arctan2(vy, -vx))
                    self.height = length
                    theta = np.deg2rad(self.angle)
                    self.u_dir[:] = [np.cos(theta), np.sin(theta)]
                    self.v_dir[:] = [-np.sin(theta), np.cos(theta)]
                    
            self.update_rectangle()

    def on_scroll(self, event):
        if self.mode != 'init' or event.inaxes != self.ax or event.xdata is None: return
        step = event.step if hasattr(event, 'step') else (1 if event.button == 'up' else -1)
        self.angle = (self.angle + 2 * step) % 360
        theta = np.deg2rad(self.angle)
        self.u_dir[:] = [np.cos(theta), np.sin(theta)]
        self.v_dir[:] = [-np.sin(theta), np.cos(theta)]
        self.update_cross(event.xdata, event.ydata)

    def on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None: return
        if self.mode == 'init':
            self.p0 = np.array([event.xdata, event.ydata])
            self.mode = 'drawing'
            self.cross1.set_visible(False)
            self.cross2.set_visible(False)
        elif self.mode == 'drawing':
            self.mode = 'edit'
            self.handles.set_visible(True)

    def on_press(self, event):
        if self.mode == 'edit' and event.inaxes == self.ax and event.xdata is not None:
            pts = self.handles.get_offsets()
            if len(pts) > 0:
                dists = np.hypot(pts[:, 0] - event.xdata, pts[:, 1] - event.ydata)
                idx = dists.argmin()
                if dists[idx] < 10:
                    self.dragging = 'corner1' if idx == 0 else 'corner2'

    def on_release(self, event):
        self.dragging = None

    def on_add_another(self, event):
        self.save_current_annotation()

    def on_undo(self, event):
        if self.completed_annotations:
            self.completed_annotations.pop()
            if self.completed_polygons:
                poly = self.completed_polygons.pop()
                poly.remove()
                self.fig.canvas.draw_idle()

    def on_finish(self, event):
        if self.mode == 'edit' and self.p0 is not None:
            self.save_current_annotation()
        self.finished = True
        plt.close(self.fig)

    def run(self):
        plt.show()

        if self.multiple:
            if self.normalize:
                normalized = []
                for corners, w, h, a in self.completed_annotations:
                    norm_c = [(x / self.img_w, y / self.img_h) for x, y in corners]
                    normalized.append((norm_c, w / self.img_w, h / self.img_h, a))
                return normalized
            return self.completed_annotations
        else:
            # original single behavior format
            if not self.completed_annotations:
                return [], 0, 0, 0
                
            last_annotation = self.completed_annotations[-1]
            corners, w, h, a = last_annotation
            if self.normalize:
                corners = [(x / self.img_w, y / self.img_h) for x, y in corners]
                w /= self.img_w
                h /= self.img_h
                
            return corners, w, h, a


class BoundingBoxVisualizer:
    """Utility class to visualize bounding boxes on images."""
    
    @staticmethod
    def show(image: typing.Union[str, Path, np.ndarray], corners_list: typing.List, multiple: bool = False):
        if isinstance(image, (str, Path)):
            img = plt.imread(str(image))
        else:
            img = image

        img_h, img_w = img.shape[0], img.shape[1]
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.imshow(img, origin='upper')

        if not multiple and (len(corners_list) > 0 and isinstance(corners_list[0], tuple) and len(corners_list[0]) == 2):
            print("Single bounding box:", corners_list)
            poly = Polygon(corners_list, closed=True, fill=False, edgecolor='lime', lw=2)
            ax.add_patch(poly)
        else:
            print(f"Multiple bounding boxes: {len(corners_list)} boxes")
            colors = ['lime', 'red', 'blue', 'yellow', 'cyan', 'magenta']
            for i, corners in enumerate(corners_list):
                if not corners:
                    continue
                color = colors[i % len(colors)]
                poly = Polygon(corners, closed=True, fill=False, edgecolor=color, lw=2)
                ax.add_patch(poly)

        ax.set_xlim(0, img_w)
        ax.set_ylim(img_h, 0)
        ax.set_aspect('equal')
        plt.show()

    @staticmethod
    def show_from_annotations(image: typing.Union[str, Path, np.ndarray], annotations: typing.List):
        corners_list = [corners for corners, _, _, _ in annotations]
        BoundingBoxVisualizer.show(image, corners_list, multiple=True)