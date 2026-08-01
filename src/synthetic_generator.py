"""
SpatioTemporal Synthetic Data Generator Module
===============================================
Generates high-fidelity synthetic spatiotemporal urban incident datasets.
Preserves spatial hotspots, temporal cycles, category imbalances, status ratios,
and weather/festival correlations. Includes adjacency-aware spatial propagation,
data augmentations, and distribution validation.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta

logger = logging.getLogger(__name__)

class SpatioTemporalSyntheticGenerator:
    """
    Stateful generator for spatiotemporal complaints datasets.
    Learns joint and conditional probability distributions from real data
    and synthesizes high-fidelity, GNN+LSTM-compatible datasets.
    """
    def __init__(self, random_seed: int = 42):
        self.seed = random_seed
        self.rng = np.random.default_rng(self.seed)
        
        # Prior parameters to learn
        self.spatial_bounds = {}
        self.zone_centroids = {}
        self.zone_stds = {}
        self.zone_probs = None
        
        self.temporal_hours_prob = None
        self.temporal_weekdays_prob = None
        self.temporal_months_prob = None
        self.base_window_rate = 0.0
        
        self.category_probs = None
        self.zone_category_probs = {}  # P(category | zone)
        self.category_metadata = {}
        self.category_status_prob = {}  # P(Open | category)
        
        self.weather_multipliers = {
            "rain_none": 1.0,
            "rain_light": 1.0,
            "rain_heavy": 1.0,
            "temp_slope": 0.0,
            "humidity_slope": 0.0
        }
        self.festival_multipliers = {
            "festival": 1.0,
            "festival_eve": 1.0
        }
        
        self.is_fitted = False

    def fit(self, df: pd.DataFrame, weather_df: pd.DataFrame = None, festivals_df: pd.DataFrame = None):
        """
        Learn the empirical distributions and prior relationships from original data.
        """
        logger.info("Fitting synthetic generator on original dataset...")
        df = df.copy()
        
        # Check required columns
        required_cols = ["created_at", "latitude", "longitude", "category_id", "complaint_status_title"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' is missing from complaints DataFrame")

        # ────── Phase 1: Spatial Patterns ──────
        logger.info("Learning spatial hotspot patterns...")
        self.spatial_bounds = {
            "min_lat": float(df["latitude"].min()),
            "max_lat": float(df["latitude"].max()),
            "min_lon": float(df["longitude"].min()),
            "max_lon": float(df["longitude"].max())
        }
        logger.info(f"City boundaries: Lat [{self.spatial_bounds['min_lat']:.4f}, {self.spatial_bounds['max_lat']:.4f}], "
                    f"Lon [{self.spatial_bounds['min_lon']:.4f}, {self.spatial_bounds['max_lon']:.4f}]")

        # Fallback zone clustering if zone_id not present
        if "zone_id" not in df.columns:
            logger.warning("zone_id column not found in training data. Clustering coordinates on the fly...")
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=12, random_state=self.seed, n_init=10)
            df["zone_id"] = kmeans.fit_predict(df[["latitude", "longitude"]])
        
        num_zones = df["zone_id"].nunique()
        zone_counts = df["zone_id"].value_counts(normalize=True).to_dict()
        self.zone_probs = np.zeros(num_zones)
        for z, p in zone_counts.items():
            if z < num_zones:
                self.zone_probs[z] = p

        for z in range(num_zones):
            z_data = df[df["zone_id"] == z]
            if len(z_data) > 0:
                self.zone_centroids[z] = (float(z_data["latitude"].mean()), float(z_data["longitude"].mean()))
                self.zone_stds[z] = (max(float(z_data["latitude"].std()), 1e-4), max(float(z_data["longitude"].std()), 1e-4))
            else:
                self.zone_centroids[z] = (df["latitude"].mean(), df["longitude"].mean())
                self.zone_stds[z] = (1e-3, 1e-3)

        # ────── Phase 2: Temporal Patterns ──────
        logger.info("Learning temporal cyclic priors...")
        df["hour_of_day"] = df["created_at"].dt.hour
        df["day_of_week"] = df["created_at"].dt.dayofweek
        df["month"] = df["created_at"].dt.month

        # Pad to full temporal ranges to prevent index out of bounds on small datasets
        hour_counts = df["hour_of_day"].value_counts(normalize=True).to_dict()
        self.temporal_hours_prob = np.array([hour_counts.get(h, 0.0) for h in range(24)])
        self.temporal_hours_prob = (self.temporal_hours_prob + 1e-6) / (self.temporal_hours_prob + 1e-6).sum()

        weekday_counts = df["day_of_week"].value_counts(normalize=True).to_dict()
        self.temporal_weekdays_prob = np.array([weekday_counts.get(w, 0.0) for w in range(7)])
        self.temporal_weekdays_prob = (self.temporal_weekdays_prob + 1e-6) / (self.temporal_weekdays_prob + 1e-6).sum()

        month_counts = df["month"].value_counts(normalize=True).to_dict()
        self.temporal_months_prob = np.array([month_counts.get(m, 0.0) for m in range(1, 13)])
        self.temporal_months_prob = (self.temporal_months_prob + 1e-6) / (self.temporal_months_prob + 1e-6).sum()

        # Base arrival rate per 6-hour window (overall mean)
        # Group by 6h window to compute historical counts
        df["time_window"] = df["created_at"].dt.floor("6h")
        window_counts = df.groupby("time_window").size()
        self.base_window_rate = float(window_counts.mean())
        logger.info(f"Learned baseline rate of {self.base_window_rate:.2f} complaints per 6-hour window")

        # ────── Phase 3: Category & Status distributions ──────
        logger.info("Learning category frequencies and resolution distributions...")
        # Category global probabilities
        all_categories = sorted(df["category_id"].unique())
        cat_counts = df["category_id"].value_counts(normalize=True).to_dict()
        self.category_probs = {c: cat_counts.get(c, 0.0) for c in all_categories}

        # Zone-specific category distribution to preserve spatial-category behavior
        for z in range(num_zones):
            z_df = df[df["zone_id"] == z]
            if len(z_df) > 0:
                z_cat_counts = z_df["category_id"].value_counts(normalize=True).to_dict()
                self.zone_category_probs[z] = {c: z_cat_counts.get(c, 0.0) for c in all_categories}
            else:
                self.zone_category_probs[z] = self.category_probs.copy()

        # Build descriptive metadata lookup map for high-fidelity rendering
        # Pick one representative real row for each (zone_id, category_id) to copy textual attributes later
        metadata_cols = [
            "sub_category_id", "civic_agency_id", "location", "address", 
            "ward_title", "category_title", "sub_category_title", 
            "civic_agency_title", "comment_count", "ward_id"
        ]
        available_metadata = [c for c in metadata_cols if c in df.columns]
        
        for (z, c), group in df.groupby(["zone_id", "category_id"]):
            rep_row = group.iloc[0]
            self.category_metadata[(z, c)] = {col: rep_row[col] for col in available_metadata}

        # Category status unresolved ratios
        # Map Open statuses as unresolved, Resolved as terminal
        open_list = ["open", "on-the-job", "re-opened"]
        df["is_unresolved"] = df["complaint_status_title"].str.strip().str.lower().isin(open_list).astype(int)
        
        for c, group in df.groupby("category_id"):
            unres_ratio = float(group["is_unresolved"].mean())
            self.category_status_prob[c] = unres_ratio

        # ────── Phase 4: Weather & Festival Conditioning ──────
        logger.info("Learning weather & festival conditional multipliers...")
        if weather_df is not None or festivals_df is not None:
            # Reconstruct daily complaints to compute correlations
            df["date"] = df["created_at"].dt.date
            daily_complaints = df.groupby("date").size().rename("complaints_count").reset_index()
            daily_complaints["date"] = pd.to_datetime(daily_complaints["date"]).dt.date
            
            # Merge weather
            if weather_df is not None:
                weather_df = weather_df.copy()
                weather_df["date"] = pd.to_datetime(weather_df["date"]).dt.date
                daily_complaints = daily_complaints.merge(weather_df, on="date", how="left")
                
                # Empirical rain multipliers
                no_rain_mean = daily_complaints[daily_complaints["rainfall"] == 0]["complaints_count"].mean()
                if pd.isna(no_rain_mean) or no_rain_mean == 0:
                    no_rain_mean = self.base_window_rate * 4.0
                
                light_rain_mean = daily_complaints[(daily_complaints["rainfall"] > 0) & (daily_complaints["rainfall"] < 5.0)]["complaints_count"].mean()
                heavy_rain_mean = daily_complaints[daily_complaints["rainfall"] >= 5.0]["complaints_count"].mean()
                
                self.weather_multipliers["rain_none"] = 1.0
                self.weather_multipliers["rain_light"] = float(light_rain_mean / no_rain_mean) if not pd.isna(light_rain_mean) else 1.1
                self.weather_multipliers["rain_heavy"] = float(heavy_rain_mean / no_rain_mean) if not pd.isna(heavy_rain_mean) else 1.4
                
                # Simple correlation slopes for temp/humidity
                # Normalizing them using covariance
                cov_temp = daily_complaints[["complaints_count", "temperature"]].cov().iloc[0, 1]
                var_temp = daily_complaints["temperature"].var()
                if var_temp > 0 and not pd.isna(cov_temp):
                    self.weather_multipliers["temp_slope"] = float(cov_temp / var_temp)
                
                cov_hum = daily_complaints[["complaints_count", "humidity"]].cov().iloc[0, 1]
                var_hum = daily_complaints["humidity"].var()
                if var_hum > 0 and not pd.isna(cov_hum):
                    self.weather_multipliers["humidity_slope"] = float(cov_hum / var_hum)
                
                logger.info(f"Weather multipliers: LightRain={self.weather_multipliers['rain_light']:.2f}, "
                            f"HeavyRain={self.weather_multipliers['rain_heavy']:.2f}, TempSlope={self.weather_multipliers['temp_slope']:.4f}")

            # Merge festivals
            if festivals_df is not None:
                festivals_df = festivals_df.copy()
                festivals_df["date"] = pd.to_datetime(festivals_df["date"]).dt.date
                daily_complaints = daily_complaints.merge(festivals_df, on="date", how="left")
                daily_complaints["festival_flag"] = daily_complaints["festival_flag"].fillna(0)
                
                fest_mean = daily_complaints[daily_complaints["festival_flag"] == 1]["complaints_count"].mean()
                norm_mean = daily_complaints[daily_complaints["festival_flag"] == 0]["complaints_count"].mean()
                
                if norm_mean > 0 and not pd.isna(fest_mean):
                    self.festival_multipliers["festival"] = float(fest_mean / norm_mean)
                else:
                    self.festival_multipliers["festival"] = 1.3  # fallback 30% surge
                
                # Festival eve surge fallback
                self.festival_multipliers["festival_eve"] = 1.15
                
                logger.info(f"Festival multiplier: {self.festival_multipliers['festival']:.2f}x surge")

        self.is_fitted = True
        logger.info("Synthetic generator fitted successfully.")

    def generate(
        self,
        start_date: str,
        end_date: str,
        target_records: int = 200000,
        adjacency_matrix: np.ndarray = None,
        spatial_smoothing_eta: float = 0.15,
        temporal_augmentation: bool = True,
        spatial_augmentation: bool = True,
        behavioral_augmentation: bool = True,
        weather_df: pd.DataFrame = None,
        festivals_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        Generate the expanded synthetic complaints dataset.
        """
        if not self.is_fitted:
            raise RuntimeError("Generator must be fitted with fit() before calling generate()")

        logger.info(f"Generating synthetic complaints from {start_date} to {end_date}...")
        
        # ────── Step 1: Create Timeline and Daily Weather/Festivals ──────
        start_ts = pd.to_datetime(start_date)
        end_ts = pd.to_datetime(end_date)
        days_diff = (end_ts - start_ts).days + 1
        
        # Timeline days
        dates_list = [start_ts + timedelta(days=i) for i in range(days_diff)]
        
        # Tiling or sampling daily weather and festivals
        synth_daily = pd.DataFrame({"date": [d.date() for d in dates_list]})
        
        if festivals_df is not None:
            # Repeatedly tile festivals matching month and day of year, or join on original festival calendar
            fest_copy = festivals_df.copy()
            fest_copy["month_day"] = pd.to_datetime(fest_copy["date"]).dt.strftime("%m-%d")
            synth_daily["month_day"] = pd.to_datetime(synth_daily["date"]).dt.strftime("%m-%d")
            # Map festival flags
            fest_map = fest_copy.groupby("month_day")["festival_flag"].max().to_dict()
            synth_daily["festival_flag"] = synth_daily["month_day"].map(fest_map).fillna(0).astype(int)
            synth_daily.drop(columns=["month_day"], inplace=True)
        else:
            synth_daily["festival_flag"] = 0

        if weather_df is not None:
            # We tile/repeat weather data to cover the synthetic years, aligning by month-day
            weather_copy = weather_df.copy()
            weather_copy["month_day"] = pd.to_datetime(weather_copy["date"]).dt.strftime("%m-%d")
            synth_daily["month_day"] = pd.to_datetime(synth_daily["date"]).dt.strftime("%m-%d")
            
            # Map mean temperature/rainfall/humidity per month-day to preserve perfect seasonal cycles
            weather_map = weather_copy.groupby("month_day").mean(numeric_only=True).to_dict()
            
            synth_daily["temperature"] = synth_daily["month_day"].map(weather_map.get("temperature", {})).fillna(24.0)
            synth_daily["rainfall"] = synth_daily["month_day"].map(weather_map.get("rainfall", {})).fillna(0.0)
            synth_daily["humidity"] = synth_daily["month_day"].map(weather_map.get("humidity", {})).fillna(60.0)
            synth_daily.drop(columns=["month_day"], inplace=True)
            
            # Add small random daily perturbations to simulated weather
            synth_daily["temperature"] += self.rng.normal(0, 1.5, size=len(synth_daily))
            synth_daily["rainfall"] = synth_daily["rainfall"].apply(lambda r: max(0, r + self.rng.normal(0, 1.0)) if r > 0 else (0.0 if self.rng.random() > 0.1 else self.rng.exponential(2.0)))
            synth_daily["humidity"] = synth_daily["humidity"].apply(lambda h: np.clip(h + self.rng.normal(0, 5.0), 20, 100))
        else:
            synth_daily["temperature"] = 25.0
            synth_daily["rainfall"] = 0.0
            synth_daily["humidity"] = 65.0

        # Calculate is_festival_eve
        festival_dates = set(synth_daily[synth_daily["festival_flag"] == 1]["date"])
        synth_daily["is_festival_eve"] = synth_daily["date"].apply(
            lambda d: 1 if (d + timedelta(days=1)) in festival_dates else 0
        )

        # ────── Step 2: Set up 6-hour Windows ──────
        num_zones = len(self.zone_centroids)
        records = []
        
        # Calculate daily multipliers
        synth_daily["weather_mult"] = 1.0
        if "rainfall" in synth_daily.columns:
            synth_daily["weather_mult"] = synth_daily["rainfall"].apply(
                lambda r: self.weather_multipliers["rain_heavy"] if r >= 5.0 
                else (self.weather_multipliers["rain_light"] if r > 0 else 1.0)
            )
            # Add temp and humidity slope adjustments
            temp_mean = synth_daily["temperature"].mean()
            hum_mean = synth_daily["humidity"].mean()
            synth_daily["weather_mult"] *= (
                1.0 + self.weather_multipliers["temp_slope"] * (synth_daily["temperature"] - temp_mean) / 100.0
            ) * (
                1.0 + self.weather_multipliers["humidity_slope"] * (synth_daily["humidity"] - hum_mean) / 100.0
            )

        synth_daily["fest_mult"] = 1.0
        synth_daily.loc[synth_daily["festival_flag"] == 1, "fest_mult"] = self.festival_multipliers["festival"]
        synth_daily.loc[synth_daily["is_festival_eve"] == 1, "fest_mult"] = self.festival_multipliers["festival_eve"]

        # If adjacency matrix exists, normalize it for spatial propagation smoothing
        if adjacency_matrix is not None:
            # Introduce asymmetric directional flow (favor South-bound spillovers along lat gradient)
            asym_adj = adjacency_matrix.copy().astype(np.float64)
            for u in range(num_zones):
                for v in range(num_zones):
                    if asym_adj[u, v] > 0:
                        lat_diff = self.zone_centroids[u][0] - self.zone_centroids[v][0]
                        if lat_diff > 0:
                            asym_adj[u, v] *= 1.5  # favor flow going South (declining latitude)
                        else:
                            asym_adj[u, v] *= 0.5  # reduce flow going North
            
            # Row-normalize to keep rate scales consistent
            row_sums = asym_adj.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            norm_adj = asym_adj / row_sums
        else:
            norm_adj = None

        # Behavioral state tracker for recurrence augmentation
        zone_unresolved_queues = {z: 0 for z in range(num_zones)}

        # Initialize decay rates per zone (for non-stationary backlog decay)
        zone_decay_rates = {z: 0.7 for z in range(num_zones)}

        # Initialize zone probabilities and centroids (for dynamic hotspot migration)
        current_zone_probs = self.zone_probs.copy()
        current_centroids = {z: list(self.zone_centroids[z]) for z in range(num_zones)}

        logger.info("Simulating spatiotemporal incident processes...")
        
        # Loop through each day and each 6-hour interval
        for _, day_row in synth_daily.iterrows():
            current_date = day_row["date"]
            
            # Dynamic Hotspot Migration: daily drift of zone probabilities
            prob_drift = self.rng.normal(0, 0.002, size=num_zones)
            current_zone_probs = np.clip(current_zone_probs + prob_drift, 0.01, 1.0)
            current_zone_probs = current_zone_probs / current_zone_probs.sum()

            # Drift zone centroids
            for z in range(num_zones):
                lat_drift = self.rng.normal(0, 0.00005)
                lon_drift = self.rng.normal(0, 0.00005)
                current_centroids[z][0] += lat_drift
                current_centroids[z][1] += lon_drift

            # Non-stationary backlog decay: fluctuate decay rates daily per zone
            for z in range(num_zones):
                zone_decay_rates[z] = float(np.clip(zone_decay_rates[z] + self.rng.normal(0, 0.02), 0.5, 0.9))

            # Weather and festival multipliers for the day
            w_mult = day_row["weather_mult"]
            f_mult = day_row["fest_mult"]
            
            # Seasonal amplification (Augmentation)
            if spatial_augmentation and day_row["rainfall"] >= 5.0:
                # scale up monsoon multipliers during heavy rain
                w_mult *= 1.2

            for hour in [0, 6, 12, 18]:
                window_dt = pd.to_datetime(f"{current_date} {hour:02d}:00:00")
                
                # Hour, weekday, month multipliers based on hourly cyclic priors
                h_prob = self.temporal_hours_prob[hour] * 24.0  # normalize
                w_prob = self.temporal_weekdays_prob[window_dt.dayofweek] * 7.0
                m_prob = self.temporal_months_prob[window_dt.month - 1] * 12.0
                
                cyclic_multiplier = h_prob * w_prob * m_prob
                
                # Calculate raw rate lambda for each zone
                # Lambda_z = base_rate * P(zone) * cyclic * weather * festival
                # Since rate is per window, overall sum of P(zone) = 1.0, so sum of lambdas = rate * cyclic * weather * festival
                rate_mult = cyclic_multiplier * w_mult * f_mult
                lambdas = self.base_window_rate * current_zone_probs * rate_mult
                
                # Behavioral recurrence simulation (Augmentation)
                if behavioral_augmentation:
                    for z in range(num_zones):
                        # If a zone has unresolved backlog, rate increases by a percentage (recurrence / duplicate reporting)
                        recurrence_boost = min(0.3, zone_unresolved_queues[z] * 0.05)
                        lambdas[z] *= (1.0 + recurrence_boost)

                # Adjacency-Aware Spatial Propagation (Phase 7)
                if norm_adj is not None:
                    # propagate eta% of the rates spatially to neighboring zones
                    lambdas = (1.0 - spatial_smoothing_eta) * lambdas + spatial_smoothing_eta * (norm_adj @ lambdas)

                # Sample actual counts using Poisson distribution
                # If target_records is much larger than original, we scale up base rate
                scale_factor = target_records / 16000.0  # ratio to achieve target expansion
                scaled_lambdas = lambdas * scale_factor
                
                counts = self.rng.poisson(scaled_lambdas)
                
                # Local burst injection (Augmentation) with cascading neighborhood propagation
                if temporal_augmentation and self.rng.random() < 0.005:
                    burst_zone = self.rng.choice(num_zones)
                    burst_size = self.rng.integers(15, 45)
                    counts[burst_zone] += burst_size
                    
                    # Cascading surge to GNN neighbors
                    if adjacency_matrix is not None:
                        neighbors = [j for j in range(num_zones) if adjacency_matrix[burst_zone, j] > 0]
                        for n_idx in neighbors:
                            counts[n_idx] += int(burst_size * 0.3)
                    logger.info(f"Augmentation: Injected cascading burst of {burst_size} complaints in Zone {burst_zone} propagating to neighbors at {window_dt}")

                # Update unresolved backlog count for the next window (tracked per zone)
                new_unresolved = {z: 0 for z in range(num_zones)}

                for z in range(num_zones):
                    z_count = counts[z]
                    if z_count <= 0:
                        continue
                    
                    # ────── Step 3: Synthesis of individual records ──────
                    # Sample Category based on zone-specific priors
                    zone_cats = list(self.zone_category_probs[z].keys())
                    zone_cat_p = list(self.zone_category_probs[z].values())
                    # Ensure sum to 1
                    zone_cat_p = np.array(zone_cat_p)
                    if zone_cat_p.sum() == 0:
                        zone_cat_p = np.array(list(self.category_probs.values()))
                    zone_cat_p = zone_cat_p / zone_cat_p.sum()
                    
                    sampled_cats = self.rng.choice(zone_cats, size=z_count, p=zone_cat_p)
                    
                    # Coordinate generation using Gaussian sampling (Phase 2)
                    mean_lat, mean_lon = current_centroids[z][0], current_centroids[z][1]
                    std_lat, std_lon = self.zone_stds[z]
                    
                    # Hotspot expansion (Augmentation)
                    if spatial_augmentation:
                        # expand coordinate variances slightly in dense hotspots
                        std_lat *= 1.15
                        std_lon *= 1.15
                        
                    lats = self.rng.normal(mean_lat, std_lat, size=z_count)
                    lons = self.rng.normal(mean_lon, std_lon, size=z_count)
                    
                    # Coordinate Jittering (Augmentation)
                    if spatial_augmentation:
                        lats += self.rng.normal(0, 0.0002, size=z_count)
                        lons += self.rng.normal(0, 0.0002, size=z_count)
                    
                    # Clip to absolute boundary box
                    lats = np.clip(lats, self.spatial_bounds["min_lat"], self.spatial_bounds["max_lat"])
                    lons = np.clip(lons, self.spatial_bounds["min_lon"], self.spatial_bounds["max_lon"])
                    
                    for idx in range(z_count):
                        cat_id = int(sampled_cats[idx])
                        
                        # Timestamp Generation (Phase 3)
                        # Distribute timestamps uniformly within the 6-hour interval
                        offset_minutes = self.rng.integers(0, 360)
                        exact_time = window_dt + timedelta(minutes=int(offset_minutes))
                        
                        # Category-specific status (Phase 4)
                        unres_p = self.category_status_prob.get(cat_id, 0.25)
                        
                        # Delayed resolution simulation (Augmentation)
                        if behavioral_augmentation and day_row["rainfall"] >= 5.0:
                            # Increase unresolved probability under extreme rain conditions
                            unres_p = min(0.9, unres_p * 1.3)
                            
                        is_open = self.rng.random() < unres_p
                        status = "Open" if is_open else "Resolved"
                        if is_open:
                            new_unresolved[z] += 1
                        
                        # Populate descriptions and details from historical lookup
                        meta = self.category_metadata.get((z, cat_id), {})
                        if not meta:
                            # Search globally for this category
                            meta = next((v for k, v in self.category_metadata.items() if k[1] == cat_id), {})
                        
                        # Defaults if lookup failed
                        sub_cat_id = meta.get("sub_category_id", 0)
                        civic_agency_id = meta.get("civic_agency_id", 0)
                        location = meta.get("location", "Unknown Location")
                        address = meta.get("address", "Unknown Address")
                        ward_title = meta.get("ward_title", "Unknown Ward")
                        category_title = meta.get("category_title", "Others")
                        sub_category_title = meta.get("sub_category_title", "Others")
                        civic_agency_title = meta.get("civic_agency_title", "BBMP")
                        comment_count = meta.get("comment_count", 0)
                        ward_id = meta.get("ward_id", 0)

                        records.append({
                            "created_at": exact_time.strftime("%m/%d/%Y %H:%M"),
                            "ward_id": ward_id,
                            "title": f"Synthetic {category_title} Complaint",
                            "description": f"Synthesized report for {sub_category_title} in zone {z}.",
                            "sub_category_id": sub_cat_id,
                            "civic_agency_id": civic_agency_id,
                            "location": location,
                            "address": address,
                            "latitude": float(lats[idx]),
                            "longitude": float(lons[idx]),
                            "ward_title": ward_title,
                            "category_id": cat_id,
                            "category_title": category_title,
                            "sub_category_title": sub_category_title,
                            "civic_agency_title": civic_agency_title,
                            "complaint_status_title": status,
                            "comment_count": comment_count,
                            "zone_id": z,
                            "temperature": float(day_row["temperature"]),
                            "rainfall": float(day_row["rainfall"]),
                            "humidity": float(day_row["humidity"]),
                            "festival_flag": int(day_row["festival_flag"])
                        })
                
                # Decay old queue and add new unresolved issues per zone using non-stationary rates
                for z in range(num_zones):
                    zone_unresolved_queues[z] = int(zone_unresolved_queues[z] * zone_decay_rates[z] + new_unresolved[z])

        synth_df = pd.DataFrame(records)
        
        # Shuffle records to simulate random reporting order
        synth_df = synth_df.sample(frac=1.0, random_state=self.seed).reset_index(drop=True)
        
        logger.info(f"Successfully generated {len(synth_df)} synthetic complaint records.")
        return synth_df

    def validate(self, real_df: pd.DataFrame, synth_df: pd.DataFrame) -> dict:
        """
        Perform statistical comparisons of spatial, temporal, and category distributions
        between real and synthetic datasets using Wasserstein distance and KL divergence.
        """
        logger.info("Validating synthetic dataset against original data...")
        real_df = real_df.copy()
        synth_df = synth_df.copy()

        # Extract distributions
        real_df["hour"] = pd.to_datetime(real_df["created_at"], format="mixed").dt.hour
        real_df["weekday"] = pd.to_datetime(real_df["created_at"], format="mixed").dt.dayofweek
        real_df["month"] = pd.to_datetime(real_df["created_at"], format="mixed").dt.month

        synth_df["hour"] = pd.to_datetime(synth_df["created_at"], format="mixed").dt.hour
        synth_df["weekday"] = pd.to_datetime(synth_df["created_at"], format="mixed").dt.dayofweek
        synth_df["month"] = pd.to_datetime(synth_df["created_at"], format="mixed").dt.month

        def get_pmf(series, categories):
            counts = series.value_counts().to_dict()
            pmf = np.array([counts.get(c, 0) for c in categories], dtype=float)
            pmf_sum = pmf.sum()
            return pmf / pmf_sum if pmf_sum > 0 else pmf

        def kl_divergence(p, q, eps=1e-10):
            p = np.clip(p, eps, 1.0)
            q = np.clip(q, eps, 1.0)
            return float(np.sum(p * np.log(p / q)))

        def wasserstein_1d_empirical(u, v):
            # Sort-based empirical 1D Wasserstein distance
            u_sorted = np.sort(u)
            v_sorted = np.sort(v)
            # Interpolate to compare same sizes
            if len(u_sorted) != len(v_sorted):
                x_u = np.linspace(0, 1, len(u_sorted))
                x_v = np.linspace(0, 1, len(v_sorted))
                v_interp = np.interp(x_u, x_v, v_sorted)
                return float(np.mean(np.abs(u_sorted - v_interp)))
            return float(np.mean(np.abs(u_sorted - v_sorted)))

        # ────── 1. Spatial Hotspots ──────
        ws_lat = wasserstein_1d_empirical(real_df["latitude"].values, synth_df["latitude"].values)
        ws_lon = wasserstein_1d_empirical(real_df["longitude"].values, synth_df["longitude"].values)

        # ────── 2. Temporal Distributions ──────
        p_hour_real = get_pmf(real_df["hour"], range(24))
        p_hour_synth = get_pmf(synth_df["hour"], range(24))
        kl_hour = kl_divergence(p_hour_real, p_hour_synth)

        p_week_real = get_pmf(real_df["weekday"], range(7))
        p_week_synth = get_pmf(synth_df["weekday"], range(7))
        kl_week = kl_divergence(p_week_real, p_week_synth)

        p_month_real = get_pmf(real_df["month"], range(1, 13))
        p_month_synth = get_pmf(synth_df["month"], range(1, 13))
        kl_month = kl_divergence(p_month_real, p_month_synth)

        # ────── 3. Categories ──────
        all_cats = sorted(list(set(real_df["category_id"]).union(set(synth_df["category_id"]))))
        p_cat_real = get_pmf(real_df["category_id"], all_cats)
        p_cat_synth = get_pmf(synth_df["category_id"], all_cats)
        kl_cat = kl_divergence(p_cat_real, p_cat_synth)

        # ────── 4. Status Ratios ──────
        open_list = ["open", "on-the-job", "re-opened"]
        real_unres = real_df["complaint_status_title"].str.strip().str.lower().isin(open_list).mean()
        synth_unres = synth_df["complaint_status_title"].str.strip().str.lower().isin(open_list).mean()

        metrics = {
            "wasserstein_latitude": ws_lat,
            "wasserstein_longitude": ws_lon,
            "kl_divergence_hour_of_day": kl_hour,
            "kl_divergence_day_of_week": kl_week,
            "kl_divergence_month": kl_month,
            "kl_divergence_categories": kl_cat,
            "real_unresolved_ratio": float(real_unres),
            "synthetic_unresolved_ratio": float(synth_unres),
            "unresolved_ratio_difference": float(abs(real_unres - synth_unres))
        }

        logger.info(f"Validation complete:")
        logger.info(f"  Wasserstein distance Lat: {ws_lat:.6f}, Lon: {ws_lon:.6f}")
        logger.info(f"  KL Divergence HourOfDay: {kl_hour:.4f}, DayOfWeek: {kl_week:.4f}, Month: {kl_month:.4f}")
        logger.info(f"  KL Divergence Categories: {kl_cat:.4f}")
        logger.info(f"  Unresolved Ratios - Real: {real_unres:.4f}, Synthetic: {synth_unres:.4f}")

        return metrics

    def plot_comparisons(self, real_df: pd.DataFrame, synth_df: pd.DataFrame, save_path: str):
        """
        Generate beautiful side-by-side comparison plots and heatmaps.
        """
        logger.info(f"Saving comparison plots to {save_path}...")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        real_df = real_df.copy()
        synth_df = synth_df.copy()

        # Parse temporal fields
        real_df["hour"] = pd.to_datetime(real_df["created_at"], format="mixed").dt.hour
        real_df["weekday"] = pd.to_datetime(real_df["created_at"], format="mixed").dt.dayofweek
        
        synth_df["hour"] = pd.to_datetime(synth_df["created_at"], format="mixed").dt.hour
        synth_df["weekday"] = pd.to_datetime(synth_df["created_at"], format="mixed").dt.dayofweek

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        sns.set_theme(style="whitegrid")

        # 1. Spatial Hotspots (Heatmap scatter overlay)
        axes[0, 0].scatter(real_df["longitude"], real_df["latitude"], alpha=0.1, s=2, color="blue", label="Real")
        axes[0, 0].set_title("Original Spatial Distribution")
        axes[0, 0].set_xlabel("Longitude")
        axes[0, 0].set_ylabel("Latitude")
        axes[0, 0].legend()

        axes[0, 1].scatter(synth_df["longitude"], synth_df["latitude"], alpha=0.1, s=2, color="red", label="Synthetic")
        axes[0, 1].set_title("Synthetic Spatial Distribution")
        axes[0, 1].set_xlabel("Longitude")
        axes[0, 1].set_ylabel("Latitude")
        axes[0, 1].legend()

        # 2. Hourly patterns
        real_h = real_df["hour"].value_counts(normalize=True).sort_index()
        synth_h = synth_df["hour"].value_counts(normalize=True).sort_index()
        
        axes[1, 0].plot(real_h.index, real_h.values, marker='o', label="Real", color="blue", linewidth=2)
        axes[1, 0].plot(synth_h.index, synth_h.values, marker='s', label="Synthetic", color="red", linestyle="--", linewidth=2)
        axes[1, 0].set_title("Hourly Cyclic Pattern Comparison")
        axes[1, 0].set_xlabel("Hour of Day")
        axes[1, 0].set_ylabel("Proportion")
        axes[1, 0].set_xticks(range(0, 24, 2))
        axes[1, 0].legend()

        # 3. Weekly patterns
        real_w = real_df["weekday"].value_counts(normalize=True).sort_index()
        synth_w = synth_df["weekday"].value_counts(normalize=True).sort_index()
        weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        
        x = np.arange(len(weekdays))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, real_w.values, width, label="Real", color="blue", alpha=0.7)
        axes[1, 1].bar(x + width/2, synth_w.values, width, label="Synthetic", color="red", alpha=0.7)
        axes[1, 1].set_title("Weekday Distribution Comparison")
        axes[1, 1].set_xlabel("Day of Week")
        axes[1, 1].set_ylabel("Proportion")
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(weekdays)
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Comparison plots saved successfully.")
