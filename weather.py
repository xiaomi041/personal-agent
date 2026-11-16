# -*- coding: utf-8 -*-
"""天气工具（Open-Meteo，免 key）。"""
import logging

import requests
from langchain_core.tools import tool

from settings import *  # noqa: F401,F403

log = logging.getLogger("agent.weather")

def _uv_level(uv: float) -> str:
    if uv < 3:
        return "弱"
    if uv < 6:
        return "中等"
    if uv < 8:
        return "强"
    if uv < 11:
        return "很强"
    return "极强"

@tool
def get_weather(city: str) -> str:
    """查询某城市未来7天天气预报：每天含天气现象、最低/最高温、紫外线指数、风向与风速（免费 Open-Meteo 接口）。"""
    import requests

    try:
        # 1) 城市名 -> 经纬度
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "zh", "format": "json"},
            timeout=10,
        ).json()
        results = geo.get("results") or []
        if not results:
            return f"未找到城市：{city}，请检查城市名"
        loc = results[0]
        lat, lon = loc["latitude"], loc["longitude"]
        city_name = loc.get("name", city)

        # 2) 7 天逐日预报
        f = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon,
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,uv_index_max,wind_speed_10m_max,wind_direction_10m_dominant",
                "forecast_days": 7,
                "timezone": "Asia/Shanghai",
            },
            timeout=10,
        ).json()
        daily = f.get("daily", {})
        times = daily.get("time", [])
        if not times:
            return f"天气查询失败：接口未返回 {city_name} 的数据"

        lines = [f"{city_name}未来7天天气："]
        import datetime as _dt
        week = "一二三四五六日"
        for i, day in enumerate(times):
            code = daily["weather_code"][i]
            tmax = daily["temperature_2m_max"][i]
            tmin = daily["temperature_2m_min"][i]
            uv = daily["uv_index_max"][i]
            ws = daily["wind_speed_10m_max"][i]
            wd = daily["wind_direction_10m_dominant"][i]
            d = _dt.date.fromisoformat(day)
            label = ["今天", "明天", "后天"][i] if i < 3 else f"{d.month}-{d.day}"
            label += f" 周{week[d.weekday()]}"
            wind = _WIND_DIRS[int((wd + 22.5) // 45) % 8] if wd is not None else "—"
            lines.append(
                f"{label}（{day}）：{_WMO_CODE.get(code, '未知')}，"
                f"{tmin:.0f}~{tmax:.0f}℃，紫外线{uv:.0f}（{_uv_level(uv)}），"
                f"{wind}{ws:.0f}km/h"
            )
        return "\n".join(lines)
    except Exception as e:
        return f"天气查询失败：{e}"

__all__ = [n for n in list(globals()) if not n.startswith("__")]
