# System Architecture
Greece Sky and Weather Nowcast  
by Iakovos Venieris

---

# Overview

Greece Sky and Weather Nowcast is a real-time rainfall nowcasting engine designed for short-term local precipitation prediction.

The system continuously analyzes weather station measurements and produces probabilistic rainfall forecasts for multiple short-term horizons.

Unlike traditional weather forecast models that rely on large-scale numerical weather prediction systems, this engine focuses on **local atmospheric micro-signals** and adapts over time using an online learning approach.

---

# System Philosophy

Rainfall events are typically preceded by small atmospheric changes such as:

• pressure tendency  
• humidity increase  
• temperature shifts  
• solar radiation reduction  
• air saturation

The system monitors these signals continuously and detects patterns that historically preceded rainfall events.

Over time the model improves by learning from actual rainfall observations.

---

# Data Flow

The system operates in a continuous loop.

1. Weather station data is received every minute.
2. Atmospheric features are calculated.
3. The prediction model evaluates rainfall probability.
4. Results are published via MQTT.
5. When rainfall occurs, the model updates its internal parameters.

This process allows the system to continuously adapt to local climatic behavior.

---

# Input Data

The system expects weather station measurements such as:

• Atmospheric Pressure  
• Temperature  
• Relative Humidity  
• Solar Radiation  
• Rainfall measurements

These values may originate from:

• local weather stations  
• Home Assistant sensors  
• MQTT feeds

---

# Feature Engineering

Raw meteorological measurements are converted into derived features that better describe atmospheric dynamics.

The most important derived signals include:

Pressure trends  
Temperature change  
Humidity change  
Dew point spread  
Solar radiation variation

These features help identify conditions commonly associated with rainfall onset.

---

# Core Atmospheric Signals

## Pressure Trend

Pressure tendency is calculated over short time windows.

Examples:

dP_10m  
dP_30m

Falling pressure often indicates incoming low-pressure systems and precipitation.

---

## Temperature Trend

Short-term temperature variations can indicate atmospheric instability or cloud formation.

Example:

dT_10m

---

## Humidity Trend

Humidity changes provide strong signals for approaching rainfall.

Example:

dRH_10m

Rapid humidity increases are often associated with rain events.

---

## Dew Point Spread

The dew point spread measures the difference between air temperature and dew point temperature.

Example:

spread_td

Small spreads indicate saturated air and high condensation probability.

---

## Solar Radiation

Solar radiation is monitored for rapid decreases that indicate cloud thickening.

Example:

solar_mean_10m

Convective rainfall events are often preceded by solar radiation drops.

---

# Day / Night Context

Solar radiation signals are interpreted differently depending on the time of day.

The system tracks this using:

is_day flag.

---

# Prediction Engine

The prediction engine evaluates atmospheric signals and estimates rainfall probability for multiple time horizons.

Forecast windows include:

30 minutes  
60 minutes  
120 minutes  
360 minutes

For each horizon the system produces a **Probability of Precipitation (PoP)**.

---

# Rainfall Intensity Estimation

In addition to rainfall probability, the system estimates potential rainfall intensity using quantile predictions.

These include:

P10  
P50  
P90

These values represent different possible rainfall scenarios.

---

# Adaptive Learning

One of the key characteristics of the system is its adaptive learning capability.

When rainfall is detected, the system performs a training update using the atmospheric conditions that preceded the event.

Over time this allows the model to better recognize rainfall patterns specific to the local environment.

Training is cumulative and improves with additional rainfall observations.

---

# Memory Management

Weather observations are stored in memory for approximately **420 minutes**.

This window allows the system to analyze atmospheric trends over several hours while keeping resource usage low.

To reduce disk activity, model persistence occurs periodically rather than continuously.

---

# Model Persistence

The trained model can be saved automatically or manually.

A manual save command allows the user to store the current model state before restarting the system.

This ensures that accumulated training data is not lost.

---

# Hardware Design Philosophy

The system is designed to operate efficiently on low-power hardware.

Typical deployment platforms include:

• Raspberry Pi 4  
• Raspberry Pi 5  
• small Linux servers  
• mini PCs

This makes the engine suitable for home weather stations and edge computing environments.

---

# Design Goals

The primary design goals of the project are:

• local rainfall detection  
• low computational cost  
• continuous learning  
• compatibility with home automation systems  
• real-time operation

---

# Project Ownership

Greece Sky and Weather Nowcast is an original project created by:

Iakovos Venieris

All intellectual property rights belong to the author.

