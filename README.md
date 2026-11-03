# 戒烟 · 10 秒短视频

一条 10 秒、竖屏 9:16（1080×1920、30 fps）的戒烟主题短视频。画面与文案全部由程序生成，没有使用外部素材，可以放心转发。

- 成片：`out/quit-smoking-10s.mp4`（H.264 + AAC）
- 封面：`out/poster.png`（结尾字幕卡截帧）
- 生成脚本：`tools/make_quit_smoking_video.py`

## 分镜

| 时间 | 画面 | 字幕 |
| --- | --- | --- |
| 0.0–4.2s | 暗场里一支香烟亮着火星，青烟缓缓升起 | 每一口，都在燃烧时间 |
| 4.2–4.8s | 香烟被掐灭、下沉，火星熄灭，烟雾散开 | — |
| 4.8–7.4s | 清晨光线升起，烟尘变成浮尘 | 掐灭它，只需要 10 秒 / 给肺一次重新呼吸的机会 |
| 7.4–10.0s | 收尾字幕卡 | 戒烟，从这一刻开始 / 把它，转给想戒烟的人 |

## 重新生成

需要 Python 3.10+、Pillow、numpy，以及 PATH 里可用的 ffmpeg：

```bash
python tools/make_quit_smoking_video.py --out out/quit-smoking-10s.mp4 --poster out/poster.png
```

常用参数都在脚本顶部（`W`/`H`、`FPS`、`DURATION`）和 `render()` 里的 `draw_caption` 调用中，改文案、时长、配色直接改完重跑即可；`--stills <目录>` 可以先只渲染几张关键帧看效果。

## 截图

![title card](out/poster.png)
