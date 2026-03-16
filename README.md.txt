# ARC Streamlit

Starter Streamlit application for the ARC hydrokinetic energy capture boat.

This app is a first-pass decision tool for Western North Carolina stream deployment screening. It uses:

- manual latitude
- manual longitude
- manual water depth
- USGS online hydro lookups when available
- regional-curve equations for the NC mountain region

## Features

- input coordinates and depth manually
- attempt USGS hydro-context lookup
- compute regional-curve bankfull estimates
- return deploy / maybe / do not deploy guidance
- display a Folium map
- provide a summary table for field interpretation

## Project Structure

arc-streamlit/
├── app.py
├── requirements.txt
├── README.md
├── utils/
│   ├── hydro_logic.py
│   ├── usgs_lookup.py
│   └── mapping.py
└── data/
    └── config.json

## Install Locally

```bash
pip install -r requirements.txt
streamlit run app.py