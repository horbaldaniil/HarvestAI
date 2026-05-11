"""Open-Meteo integration package.

Free, no-API-key historical (1940+) and forecast weather. Daily aggregates
are the unit of work — we fetch once per field per refresh, cache to
weather_observations, and consume from there during ML feature building
and dashboard display.
"""
