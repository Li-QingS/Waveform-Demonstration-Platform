"""Two native entry buttons over a light satellite/terrestrial link schematic."""

import math

from PyQt5.QtCore import QPointF, QRect, QRectF, QSize, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QIcon, QPainter, QPainterPath, QPen, QPixmap
from PyQt5.QtWidgets import (
    QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)


BLUE = (0, 114, 189)
ORANGE = (217, 83, 25)
UI_FONT_FAMILY = "Microsoft YaHei"


class _EntryButton(QPushButton):
    """Keep Fusion's button states while giving the two text lines a hierarchy."""

    def __init__(self, title, description, name, colour, parent=None):
        super().__init__(f"{title}\n{description}", parent)
        self._title, self._description = title, description
        self._colour = QColor(*colour)
        self._hover_colour = QColor("#EDF6FC" if name == "simulation" else "#FFF4ED")
        self._pressed_colour = QColor("#DDEBF6" if name == "simulation" else "#FCE7D9")
        self.setObjectName(f"enter_{name}")
        self.setFont(QFont(UI_FONT_FAMILY, 10))
        self.setAccessibleName(f"进入{title}")
        self.setAccessibleDescription(description)
        self.setMinimumHeight(120)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setCursor(Qt.PointingHandCursor)
        self.setIcon(self._entry_icon(name, colour))
        self.setIconSize(QSize(64, 64))

    def sizeHint(self):
        return QSize(560, 124)

    def minimumSizeHint(self):
        return QSize(380, 120)

    @staticmethod
    def _entry_icon(name, colour):
        # Draw above display size so outlines remain crisp at desktop scaling.
        pixmap = QPixmap(144, 144)
        pixmap.setDevicePixelRatio(2.)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor(*colour), 2.1, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(QPen(QColor(*colour, 48), 1.0))
        painter.setBrush(QColor(*colour, 14))
        painter.drawRoundedRect(QRectF(2, 2, 68, 68), 12, 12)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(pen)
        if name == "simulation":
            painter.drawRoundedRect(QRectF(11, 14, 50, 36), 3, 3)
            painter.setPen(QPen(QColor(*colour, 95), 1.0))
            painter.drawLine(12, 22, 60, 22)
            painter.drawLine(18, 43, 55, 43)
            painter.drawLine(18, 28, 18, 43)
            painter.setBrush(QColor(*colour, 140))
            for x in (16, 20, 24):
                painter.drawEllipse(QPointF(x, 18), .8, .8)
            painter.setBrush(Qt.NoBrush)
            painter.setPen(pen)
            signal = QPainterPath(QPointF(19, 36))
            for step in range(1, 73):
                x = 19 + step / 2.
                signal.lineTo(x, 35 - 7 * math.sin((x - 19) * math.pi / 15))
            painter.drawPath(signal)
            painter.drawLine(36, 50, 36, 57)
            painter.drawLine(26, 58, 46, 58)
        else:
            top = QPainterPath(QPointF(11, 27))
            top.lineTo(18, 19); top.lineTo(55, 19); top.lineTo(61, 27)
            painter.setBrush(QColor(*colour, 9))
            painter.drawPath(top)
            painter.drawRoundedRect(QRectF(11, 27, 50, 26), 3, 3)
            painter.setBrush(Qt.NoBrush)
            for x in (23, 47):
                painter.drawEllipse(QPointF(x, 39), 4.2, 4.2)
                painter.setPen(QPen(QColor(*colour, 115), 1.0))
                painter.drawEllipse(QPointF(x, 39), 1.6, 1.6)
                painter.setPen(pen)
            painter.setPen(QPen(QColor(*colour, 115), 1.3, Qt.SolidLine, Qt.RoundCap))
            for x in (29, 34, 39):
                painter.drawLine(x, 23, x + 3, 23)
            painter.setBrush(QColor(*colour))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPointF(35, 38), 1.8, 1.8)
            painter.drawRoundedRect(QRectF(17, 54, 9, 3), 1, 1)
            painter.drawRoundedRect(QRectF(46, 54, 9, 3), 1, 1)
        painter.end()
        return QIcon(pixmap)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(self.rect()).adjusted(.5, .5, -.5, -.5)
        fill = self._pressed_colour if self.isDown() else self._hover_colour if self.underMouse() else QColor("#FFFFFF")
        shape = QPainterPath()
        shape.addRoundedRect(rect, 6, 6)
        painter.fillPath(shape, fill)
        painter.save()
        painter.setClipPath(shape)
        painter.fillRect(QRectF(rect.left(), rect.top(), 6, rect.height()), self._colour)
        painter.restore()
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor("#D1DDE8"), 1))
        painter.drawRoundedRect(rect, 6, 6)
        if self.hasFocus():
            painter.setPen(QPen(self._colour, 1.5))
            painter.drawRoundedRect(rect.adjusted(3, 3, -3, -3), 4, 4)
        shift = 1 if self.isDown() else 0
        self.icon().paint(painter, QRect(23 + shift, (self.height() - 64) // 2 + shift, 64, 64))
        left, right = 107 + shift, self.width() - 36
        text_width = max(1, right - left)
        title_font = QFont(self.font()); title_font.setPointSize(17); title_font.setWeight(QFont.DemiBold)
        painter.setFont(title_font)
        painter.setPen(QColor("#20364A"))
        painter.drawText(QRectF(left, self.height() / 2 - 29 + shift, text_width, 32),
                         Qt.AlignLeft | Qt.AlignVCenter, self._title)
        detail_font = QFont(self.font()); detail_font.setPointSize(10)
        painter.setFont(detail_font)
        painter.setPen(self.palette().color(self.palette().Disabled, self.palette().ButtonText)
                       if not self.isEnabled() else QColor("#5D7185"))
        detail = painter.fontMetrics().elidedText(self._description, Qt.ElideRight, int(text_width))
        painter.drawText(QRectF(left, self.height() / 2 + 8 + shift, text_width, 23),
                         Qt.AlignLeft | Qt.AlignVCenter, detail)
        painter.setPen(QPen(QColor(self._colour.red(), self._colour.green(), self._colour.blue(), 160),
                            1.8, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        x, y = self.width() - 24 + shift, self.height() / 2 + shift
        path = QPainterPath(QPointF(x - 4, y - 6))
        path.lineTo(x + 2, y); path.lineTo(x - 4, y + 6)
        painter.drawPath(path)


class LandingPage(QWidget):
    simulation_requested = pyqtSignal()
    hardware_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("platformEntry")
        self.setFont(QFont(UI_FONT_FAMILY, 10))
        self.setStyleSheet(
            "QWidget#platformEntry { background-color: #F3F6FA; color: #20364A; }"
            "QLabel#landingTitle { color: #18324B; }"
            "QLabel#landingSubtitle { color: #536B80; }"
            "QLabel#landingFooter { color: #64798C; }"
        )
        outer = QVBoxLayout(self)
        outer.setContentsMargins(32, 26, 32, 26)
        outer.addStretch(2)
        self._content = QWidget()
        self._content.setMaximumWidth(640)
        column = QVBoxLayout(self._content)
        column.setContentsMargins(20, 18, 20, 18)
        column.setSpacing(14)
        title = QLabel("软波形自适应演示平台")
        title.setObjectName("landingTitle")
        title.setAlignment(Qt.AlignCenter)
        title_font = QFont(self.font()); title_font.setPointSize(25); title_font.setWeight(QFont.DemiBold)
        title.setFont(title_font)
        column.addWidget(title)
        subtitle = QLabel("FDIDM / OFDM / OTFS / AFDM")
        subtitle.setObjectName("landingSubtitle")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setFont(QFont(UI_FONT_FAMILY, 11))
        column.addWidget(subtitle)
        column.addSpacing(10)
        self.simulation_button = _EntryButton(
            "仿真", "波形仿真 · 自适应分析 · 性能对比", "simulation", BLUE
        )
        self.hardware_button = _EntryButton(
            "硬件验证", "USRP 链路验证 · 实时观测 · 测评结果", "hardware", ORANGE
        )
        column.addWidget(self.simulation_button)
        column.addWidget(self.hardware_button)
        self.simulation_button.clicked.connect(self.simulation_requested)
        self.hardware_button.clicked.connect(self.hardware_requested)
        outer.addWidget(self._content, alignment=Qt.AlignHCenter)
        outer.addStretch(2)
        footer = QLabel("波形仿真与 USRP 原型验证  ·  v6.0")
        footer.setObjectName("landingFooter")
        footer.setAlignment(Qt.AlignCenter)
        footer.setFont(QFont(UI_FONT_FAMILY, 9))
        outer.addWidget(footer)

    @staticmethod
    def _pen(colour, alpha, width=1.):
        return QPen(QColor(*colour, alpha), width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)

    def _draw_satellite(self, painter, x, y, scale):
        painter.save(); painter.translate(x, y); painter.scale(scale, scale); painter.rotate(-24)
        painter.setPen(self._pen(BLUE, 112, 1.7)); painter.setBrush(QColor(*BLUE, 9))
        painter.drawRoundedRect(QRectF(-17, -15, 34, 30), 3, 3)
        painter.drawRect(QRectF(-75, -22, 46, 44)); painter.drawRect(QRectF(29, -22, 46, 44))
        painter.drawLine(-29, 0, -17, 0); painter.drawLine(17, 0, 29, 0)
        painter.setPen(self._pen(BLUE, 75, 1.))
        for side in (-1, 1):
            for dx in (40, 52, 64):
                painter.drawLine(side * dx, -21, side * dx, 21)
            painter.drawLine(side * 30, 0, side * 74, 0)
        painter.setBrush(Qt.NoBrush); painter.setPen(self._pen(BLUE, 110, 1.6))
        painter.drawLine(0, 15, 0, 27); painter.drawEllipse(QPointF(0, 29), 3, 3)
        for radius in (43, 57, 71):
            painter.drawArc(QRectF(-radius, 26 - radius, 2 * radius, 2 * radius), 230 * 16, 80 * 16)
        painter.restore()

    def _draw_station(self, painter, x, y, scale):
        painter.save(); painter.translate(x, y); painter.scale(scale, scale)
        painter.setBrush(Qt.NoBrush); painter.setPen(self._pen(BLUE, 105, 1.7))
        # A tilted reflector and feed identify a ground station.
        dish = QPainterPath(QPointF(-28, -26))
        dish.cubicTo(-26, 8, -6, 26, 25, 27)
        dish.lineTo(-28, -26)
        painter.setBrush(QColor(*BLUE, 8)); painter.drawPath(dish)
        painter.setBrush(Qt.NoBrush)
        painter.drawLine(-9, 8, 17, -17); painter.drawEllipse(QPointF(18, -18), 3, 3)
        painter.drawLine(-3, 23, -3, 48); painter.drawLine(-3, 29, -22, 52)
        painter.drawLine(-28, 53, 23, 53)
        painter.setPen(self._pen(BLUE, 65, 1.2))
        for radius in (24, 37):
            painter.drawArc(QRectF(18 - radius, -18 - radius, 2 * radius, 2 * radius), 20 * 16, 65 * 16)
        painter.restore()

    def _draw_signal_panel(self, painter, rect):
        painter.setPen(self._pen(BLUE, 55)); painter.setBrush(QColor(255, 255, 255, 75))
        painter.drawRoundedRect(rect, 5, 5)
        inset = rect.adjusted(13, 16, -13, -13)
        painter.setPen(self._pen(BLUE, 65))
        painter.drawLine(QPointF(inset.left(), inset.center().y()), QPointF(inset.right(), inset.center().y()))
        painter.drawLine(QPointF(inset.center().x(), inset.top()), QPointF(inset.center().x(), inset.bottom()))
        painter.setPen(Qt.NoPen)
        for i, q in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            for dx, dy in ((0, 0), (-2, 1), (2, -2)):
                painter.setBrush(QColor(*BLUE, 75 if dx else 120))
                painter.drawEllipse(QPointF(inset.center().x() + i * inset.width() * .26 + dx,
                                            inset.center().y() + q * inset.height() * .26 + dy), 2, 2)
        painter.setPen(self._pen(BLUE, 115))
        painter.drawText(rect.adjusted(8, 2, -8, -2), Qt.AlignTop | Qt.AlignLeft, "I / Q")

    def _draw_spectrum_panel(self, painter, rect):
        painter.setPen(self._pen(ORANGE, 55)); painter.setBrush(QColor(255, 255, 255, 75))
        painter.drawRoundedRect(rect, 5, 5)
        plot = rect.adjusted(12, 23, -12, -12)
        painter.setPen(self._pen(ORANGE, 70))
        painter.drawLine(QPointF(plot.left(), plot.bottom()), QPointF(plot.right(), plot.bottom()))
        path = QPainterPath(QPointF(plot.left(), plot.bottom() - 3))
        for step in range(81):
            t = step / 80.
            envelope = math.exp(-((t - .5) / .23)**8)
            ripple = 1. + .11 * math.sin(step * 1.7)
            path.lineTo(plot.left() + t * plot.width(), plot.bottom() - 3 - plot.height() * .78 * envelope * ripple)
        painter.setPen(self._pen(ORANGE, 115, 1.4)); painter.drawPath(path)
        painter.drawText(rect.adjusted(8, 3, -8, -3), Qt.AlignTop | Qt.AlignLeft, "RF / BASEBAND")

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        width, height = self.width(), self.height()
        painter.setPen(QPen(QColor(68, 101, 132, 18), 1))
        for x in range(0, width, 40):
            painter.drawLine(x, 0, x, height)
        for y in range(0, height, 40):
            painter.drawLine(0, y, width, y)
        # Instruments live at the edges, with the orbit above and coverage
        # below the entries. Avoid clipping whole objects at an invisible
        # centre rectangle when the window is narrowed.
        scale = min(1.35, max(.85, width / 1100.))
        font = QFont(self.font()); font.setPointSize(8); painter.setFont(font)
        painter.setBrush(Qt.NoBrush); painter.setPen(self._pen(BLUE, 47, 1.2))
        painter.drawArc(QRectF(-width * .08, height * .08, width * 1.16, height * .9), 15 * 16, 155 * 16)
        painter.drawArc(QRectF(-width * .12, height * .035, width * 1.24, height * 1.02), 15 * 16, 155 * 16)
        self._draw_satellite(painter, width * .84, height * .21, scale)
        self._draw_station(painter, width * .135, height * .66, scale)
        route = QPainterPath(QPointF(width * .83, height * .28))
        route.cubicTo(width * .96, height * .40, width * .94, height * .70, width * .82, height * .82)
        route.cubicTo(width * .67, height * .97, width * .24, height * .98, width * .135, height * .74)
        pen = self._pen(BLUE, 72, 1.2); pen.setStyle(Qt.DashLine); painter.setPen(pen)
        painter.drawPath(route)
        self._draw_signal_panel(painter, QRectF(width * .055, height * .30, width * .15, height * .15))
        self._draw_spectrum_panel(painter, QRectF(width * .80, height * .45, width * .15, height * .15))
        # Ground coverage and connected nodes occupy the lower visual layer.
        painter.setBrush(QColor(*BLUE, 4)); painter.setPen(self._pen(BLUE, 54, 1.2))
        earth = QRectF(-width * .20, height * .79, width * 1.40, height * .9)
        painter.drawEllipse(earth)
        painter.setBrush(Qt.NoBrush)
        for shrink in (.14, .29):
            painter.drawEllipse(earth.adjusted(width * shrink, 0, -width * shrink, 0))
        for offset in (.065, .13):
            painter.drawArc(earth.translated(0, height * offset), 0, 180 * 16)
        nodes = [QPointF(width * x, height * y) for x, y in ((.11, .88), (.28, .93), (.72, .93), (.89, .88))]
        painter.setPen(self._pen(BLUE, 60, 1.3))
        for first, second in zip(nodes, nodes[1:]):
            painter.drawLine(first, second)
        for index, point in enumerate(nodes):
            colour = BLUE if index % 2 == 0 else ORANGE
            painter.setPen(self._pen(colour, 88, 1.2)); painter.setBrush(QColor(250, 250, 250, 220))
            painter.drawEllipse(point, 5, 5)
            painter.setPen(Qt.NoPen); painter.setBrush(QColor(*colour, 110)); painter.drawEllipse(point, 2, 2)
        painter.setBrush(Qt.NoBrush)
        for phase, colour, baseline in ((0., BLUE, .12), (1.3, ORANGE, .145)):
            wave = QPainterPath()
            for x in range(0, width + 4, 4):
                y = height * baseline + math.sin(x / 38. + phase) * height * .016
                if x == 0:
                    wave.moveTo(x, y)
                else:
                    wave.lineTo(x, y)
            painter.setPen(self._pen(colour, 46, 1.1)); painter.drawPath(wave)
        painter.end()
