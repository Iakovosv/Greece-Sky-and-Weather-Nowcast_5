🌧️ Greece Sky and Weather Nowcast
by Iakovos Venieris

A real-time adaptive rainfall nowcasting engine designed for Greece.

This system analyzes local weather station data and produces short-term rainfall probability forecasts using an adaptive online learning model.

The engine is optimized for Home Assistant and MQTT environments and is designed to run efficiently on low-power devices such as Raspberry Pi.

⚠️ This project is not open source. The code is proprietary and protected under a restricted license.

Project Overview

Greece Sky and Weather Nowcast is a lightweight machine-learning based rainfall detection and nowcasting system.

The model continuously analyzes atmospheric changes and learns from real rainfall events to improve future predictions. Unlike traditional weather forecasts, this system focuses on very short-term local prediction (nowcasting).

Typical prediction horizons

30 minutes

60 minutes

120 minutes

360 minutes

The system updates predictions every minute and retrains itself automatically when rainfall events occur.

Key Features

Minute-level rainfall monitoring and updates

Adaptive online learning (model improves while running)

Rain probability forecasting (PoP)

Rainfall intensity quantiles (P10, P50, P90 scenarios)

Automatic feature extraction from raw station data

MQTT integration (fully Home Assistant ready)

Optimized for low-power hardware (Raspberry Pi 4 / 5)

Home Assistant Addon Installation

This project is designed to run as a Home Assistant Add-on.

Installation Steps

Open Home Assistant File Editor.

Copy the folder nowcast_ml into the Home Assistant:

/addons/

directory.

Example:

/addons/nowcast_ml

After copying the folder, go to:

Settings → Add-ons

The Greece Sky and Weather Nowcast add-on will appear in the Local Add-ons section.

Click Install and then Start the add-on.

How It Works (Machine Learning Approach)

The model analyzes short-term atmospheric changes that often precede rainfall events. When rainfall occurs, the system stores the atmospheric conditions that preceded it and retrains the model.

Initial Learning Phase

During the first deployment, the model requires rainfall events to learn local atmospheric patterns.

Accuracy improves as more events are observed:

First rainfall events: Model begins calibration

~10 rainfall events: Early pattern recognition

~30–40 rainfall events: Significantly improved prediction accuracy

Model Maturity

During the early phase, the system may produce conservative rainfall probabilities to avoid false alarms.

As it matures, it becomes more sensitive and accurate to local climate signals.

Feature Engineering

The system extracts multiple atmospheric signals from raw weather station data to identify rainfall patterns.

Pressure Trends

dP_10m / dP_30m
Pressure change (falling pressure often indicates rain)

p_mean_30m
Average pressure

Temperature Change

dT_10m
Rapid cooling can indicate cloud thickening or rain onset

Humidity Change

dRH_10m
Rising humidity frequently precedes rainfall

Dew Point Spread

spread_td
Small spread values indicate saturated air

Solar Radiation

solar_mean_10m
Sudden drops can indicate cloud formation or approaching rain cells

Day/Night Detection

is_day
Solar radiation is interpreted differently between day and night

Probabilities & Quantiles
PoP (Probability of Precipitation)

Example:

PoP 60m is the probability of measurable rainfall within the next 60 minutes.

Rainfall Quantiles (Intensity)

P10: Light rainfall scenario

P50: Median rainfall estimate

P90: High rainfall (worst-case scenario)

Debug Feature Output

The system provides debug logs to help you understand how the model interprets the current conditions.

{
 "ts": "timestamp",
 "dP_10m": "pressure trend",
 "dP_30m": "pressure trend",
 "dT_10m": "temperature change",
 "dRH_10m": "humidity change",
 "spread_td": "dew point spread",
 "p_mean_30m": "mean pressure",
 "solar_mean_10m": "solar radiation mean",
 "is_day": "day/night flag",
 "pop_60m": "rainfall probability",
 "p50_60m": "expected rainfall median"
}
Data Storage & Hardware
Memory Architecture

Weather observations are stored in RAM (~900 minutes (15 hours)) to avoid constant disk writes.

Disk Protection

To reduce wear on Raspberry Pi SD cards, the model state is saved to disk periodically (once per day).

Manual Save

Use the command Nowcast Save Model to store the trained model before a manual restart.

Compatibility

Tested on:

Raspberry Pi 4

Raspberry Pi 5

The system can run on any Linux system with Python, but it is optimized for low-power Home Assistant environments.

Support the Project If you find this project useful and want to support its development:

## ☕ Support the Project
If you find these tools valuable, you can support "Greece Sky and Weather" by buying me a coffee. Your contributions help me spend more time developing advanced tools for the community!

[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-Donate-orange?style=for-the-badge&logo=buy-me-a-coffee)](https://buymeacoffee.com/greeceskyandweather)


License

This project is proprietary software.

Use, modification, redistribution, or commercial use of this code is strictly prohibited without explicit written permission from the author.

See the LICENSE file for full details and commercial inquiries.
Contact Author: Iakovos Venieris

📩 Contact: [ greekskyweather@gmail.com ]

## 📺 YouTube Channel
Experience the sky through our lens. Join us for 24/7 live weather feeds from our stations, spectacular sky timelapses, and astronomy insights.

👉 **[Greece Sky and Weather](https://www.youtube.com/@GreeceSkyandWeather)**

*Monitoring the Greek sky and weather.*


Project: Greece Sky and Weather
