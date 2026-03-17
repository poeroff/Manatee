import importlib.util
import pathlib
import sys

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)

# ── 백엔드 import (LLM.py) ─────────────────────────────────
spec = importlib.util.spec_from_file_location(
    "agent_backend",
    pathlib.Path(__file__).parent / "LLM.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
run_agent     = mod.run_agent
clear_history = mod.clear_history


# ── 백그라운드 워커 ────────────────────────────────────────
class AgentWorker(QThread):
    finished = pyqtSignal(str)
    error    = pyqtSignal(str)

    def __init__(self, query: str):
        super().__init__()
        self.query = query

    def run(self):
        try:
            result = run_agent(self.query)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


# ── 메시지 버블 위젯 ──────────────────────────────────────
class MessageBubble(QLabel):
    def __init__(self, text: str, is_user: bool):
        super().__init__(text)
        self.setWordWrap(True)
        self.setFont(QFont("Malgun Gothic", 10))
        self.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        self.setMaximumWidth(520)

        if is_user:
            self.setStyleSheet("""
                QLabel {
                    background-color: #89b4fa;
                    color: #1e1e2e;
                    border-radius: 12px;
                    padding: 10px 14px;
                }
            """)
        else:
            self.setStyleSheet("""
                QLabel {
                    background-color: #313244;
                    color: #cdd6f4;
                    border-radius: 12px;
                    padding: 10px 14px;
                }
            """)


# ── 메인 채팅 창 ──────────────────────────────────────────
class ChatWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Manatee Agent")
        self.setMinimumSize(700, 560)
        self.worker = None
        self._build_ui()

    def _build_ui(self):
        root = QWidget()
        root.setStyleSheet("background-color: #1e1e2e;")
        self.setCentralWidget(root)

        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 타이틀바
        title_bar = QWidget()
        title_bar.setStyleSheet("background-color: #181825;")
        title_row = QHBoxLayout(title_bar)
        title_row.setContentsMargins(12, 8, 12, 8)

        title = QLabel("🧬 Manatee Agent")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont("Malgun Gothic", 13, QFont.Bold))
        title.setStyleSheet("color: #cdd6f4;")

        self.clear_btn = QPushButton("대화 초기화")
        self.clear_btn.setFont(QFont("Malgun Gothic", 9))
        self.clear_btn.setFixedWidth(90)
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background-color: #45475a;
                color: #cdd6f4;
                border-radius: 6px;
                padding: 5px;
            }
            QPushButton:hover { background-color: #585b70; }
        """)
        self.clear_btn.clicked.connect(self._clear_chat)

        title_row.addStretch()
        title_row.addWidget(title)
        title_row.addStretch()
        title_row.addWidget(self.clear_btn)
        layout.addWidget(title_bar)

        # 스크롤 영역
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("border: none; background-color: #1e1e2e;")
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.msg_container = QWidget()
        self.msg_container.setStyleSheet("background-color: #1e1e2e;")
        msg_layout = QVBoxLayout(self.msg_container)
        msg_layout.setContentsMargins(16, 16, 16, 16)
        msg_layout.setSpacing(10)
        msg_layout.addStretch()

        self.scroll.setWidget(self.msg_container)
        layout.addWidget(self.scroll, stretch=1)

        # 입력 영역
        input_bar = QWidget()
        input_bar.setStyleSheet("background-color: #181825; padding: 8px;")
        input_row = QHBoxLayout(input_bar)
        input_row.setContentsMargins(12, 8, 12, 8)
        input_row.setSpacing(8)

        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("세포 이름이나 질문을 입력하세요…")
        self.input_box.setFont(QFont("Malgun Gothic", 10))
        self.input_box.setStyleSheet("""
            QLineEdit {
                background-color: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 8px;
                padding: 8px 12px;
            }
            QLineEdit:focus { border: 1px solid #89b4fa; }
        """)
        self.input_box.returnPressed.connect(self._send)

        self.send_btn = QPushButton("전송")
        self.send_btn.setFont(QFont("Malgun Gothic", 10, QFont.Bold))
        self.send_btn.setFixedWidth(72)
        self.send_btn.setStyleSheet("""
            QPushButton {
                background-color: #89b4fa;
                color: #1e1e2e;
                border-radius: 8px;
                padding: 8px;
            }
            QPushButton:hover { background-color: #b4d0ff; }
            QPushButton:disabled { background-color: #45475a; color: #6c7086; }
        """)
        self.send_btn.clicked.connect(self._send)

        input_row.addWidget(self.input_box)
        input_row.addWidget(self.send_btn)
        layout.addWidget(input_bar)

    # ── 버블 추가 헬퍼 ────────────────────────────────────
    def _add_bubble(self, text: str, is_user: bool):
        row_widget = QWidget()
        row_widget.setStyleSheet("background-color: transparent;")
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)

        bubble = MessageBubble(text, is_user)
        if is_user:
            row_layout.addStretch()
            row_layout.addWidget(bubble)
        else:
            row_layout.addWidget(bubble)
            row_layout.addStretch()

        # stretch 앞에 삽입 (마지막 아이템이 stretch)
        layout = self.msg_container.layout()
        layout.insertWidget(layout.count() - 1, row_widget)
        QApplication.processEvents()
        self.scroll.verticalScrollBar().setValue(
            self.scroll.verticalScrollBar().maximum()
        )
        return bubble

    # ── 전송 ─────────────────────────────────────────────
    def _send(self):
        text = self.input_box.text().strip()
        if not text or self.worker is not None:
            return

        self.input_box.clear()
        self._add_bubble(text, is_user=True)

        self._set_input_enabled(False)
        self.loading_bubble = self._add_bubble("분석 중…", is_user=False)

        self.worker = AgentWorker(text)
        self.worker.finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_finished(self, result: str):
        self._remove_loading()
        self._add_bubble(result, is_user=False)
        self._set_input_enabled(True)
        self.worker = None

    def _on_error(self, err: str):
        self._remove_loading()
        self._add_bubble(f"오류 발생: {err}", is_user=False)
        self._set_input_enabled(True)
        self.worker = None

    def _remove_loading(self):
        if self.loading_bubble is not None:
            row_widget = self.loading_bubble.parent()
            if row_widget:
                row_widget.deleteLater()
            self.loading_bubble = None

    def _clear_chat(self):
        clear_history()
        # 화면의 메시지 버블 전체 제거
        layout = self.msg_container.layout()
        while layout.count() > 1:  # 마지막 stretch는 유지
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _set_input_enabled(self, enabled: bool):
        self.input_box.setEnabled(enabled)
        self.send_btn.setEnabled(enabled)
        if enabled:
            self.input_box.setFocus()


# ── 진입점 ────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = ChatWindow()
    win.show()
    sys.exit(app.exec_())
