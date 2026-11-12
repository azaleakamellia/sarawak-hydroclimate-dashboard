import ee
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ==============================================================================
# SCIENTIFIC CONFIGURATION & METADATA
# ==============================================================================
SPE_METADATA = {
    "CHIRPS": {"asset": "UCSB-CHG/CHIRPS/DAILY", "band": "precipitation", "native_scale": 5566, "factor": 1.0},
    "ERA5_LAND": {"asset": "ECMWF/ERA5_LAND/MONTHLY_AGGR", "band": "total_precipitation_sum", "native_scale": 11132, "factor": 1000.0},
    "PERSIANN": {"asset": "NOAA/PERSIANN-CDR", "band": "precipitation", "native_scale": 27830, "factor": 1.0},
}

PET_METADATA = {
    "ERA5_LAND": {"asset": "ECMWF/ERA5_LAND/MONTHLY_AGGR", "band": "potential_evaporation_sum", "native_scale": 11132, "factor": 1000.0},
}

DEM_METADATA = {
    "SRTM": {"asset": "USGS/SRTMGL1_003", "band": "elevation", "native_scale": 30, "factor": 1.0},
}

LULC_METADATA = {
    "MODIS": {"asset": "MODIS/061/MCD12Q1", "band": "LC_Type1", "native_scale": 500, "factor": 1.0},
}

def _month_bounds(year: int, month: int):
    start = datetime(year, month, 1)
    end = (datetime(year + 1, 1, 1) if month == 12 else datetime(year, month + 1, 1))
    return start, end


class HydroSampler:
    def __init__(self, gauge_csv, col_station: str, col_lat: str, col_lon: str, 
                 start_date: str, end_date: str, project_id: str, 
                 master_scale: int = 5566, sampling: str = "native"):
        if sampling not in ("native", "master"):
            raise ValueError("sampling must be 'native' or 'master'.")

        self.start_date = start_date
        self.end_date = end_date
        self.master_scale = int(master_scale)
        self.sampling = sampling

        self.start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        if self.start_dt >= self.end_dt:
            raise ValueError("start_date must precede end_date.")

        try:
            ee.Initialize(project=project_id)
            print(f"✅ GEE initialized: {project_id}")
        except Exception as exc:
            print("❌ GEE initialization failed.")
            raise

        raw = pd.read_csv(gauge_csv)
        missing = {col_station, col_lat, col_lon} - set(raw.columns)
        if missing:
            raise ValueError(f"CSV is missing columns: {sorted(missing)}")

        raw = raw.rename(columns={col_station: "station_id"})
        raw["station_id"] = raw["station_id"].astype(str)
        raw["lon"] = pd.to_numeric(raw[col_lon], errors="raise").astype(float)
        raw["lat"] = pd.to_numeric(raw[col_lat], errors="raise").astype(float)

        self.station_table = raw[["station_id", "lon", "lat"]].copy()
        gdf = gpd.GeoDataFrame(
            self.station_table,
            geometry=gpd.points_from_xy(self.station_table.lon, self.station_table.lat),
            crs="EPSG:4326",
        )
        self.gee_points = ee.FeatureCollection(gdf.__geo_interface__)
        
        self._xy_to_id = {
            (round(float(r.lon), 5), round(float(r.lat), 5)): r.station_id
            for r in self.station_table.itertuples(index=False)
        }

    def _scale_for(self, meta: dict) -> int:
        return meta["native_scale"] if self.sampling == "native" else self.master_scale

    def _year_stack(self, meta: dict, year: int):
        ic = (ee.ImageCollection(meta["asset"]).filterDate(self.start_date, self.end_date).select(meta["band"]))
        if meta["factor"] != 1.0:
            factor = meta["factor"]
            ic = ic.map(lambda img: img.multiply(factor).copyProperties(img, ["system:time_start"]))

        bands, months = [], []
        for month in range(1, 13):
            m_start, m_end = _month_bounds(year, month)
            if m_start < self.start_dt.replace(day=1) or m_start > self.end_dt:
                continue
            monthly = ic.filterDate(m_start, m_end).sum().rename(f"m{month:02d}")
            bands.append(monthly)
            months.append(pd.Timestamp(m_start))
        if not bands: return None, []
        return ee.Image.cat(bands), months

    def _sample_stack(self, stack, months, var_name: str, scale: int) -> pd.DataFrame:
        expected = [f"m{m.month:02d}" for m in months]
        actual = stack.bandNames().getInfo()
        if actual != expected:
            raise RuntimeError(f"[{var_name}] Band-name mismatch.")

        features = stack.reduceRegions(
            collection=self.gee_points, reducer=ee.Reducer.mean(), scale=scale, crs="EPSG:4326"
        ).getInfo().get("features", [])

        records = []
        for feat in features:
            props = feat.get("properties", {})
            xy = tuple(round(float(v), 5) for v in feat["geometry"]["coordinates"][:2])
            sid = props.get("station_id") or self._xy_to_id.get(xy)
            if sid is None: raise RuntimeError(f"[{var_name}] Could not identify feature: {xy}")

            for ts in months:
                value = props.get(f"m{ts.month:02d}")
                if var_name == "pet" and value is not None: value = abs(value)
                records.append({"station_id": sid, "date": ts, var_name: value})
        return pd.DataFrame(records)

    def _extract_product(self, meta: dict, var_name: str) -> pd.DataFrame:
        scale = self._scale_for(meta)
        print(f"  → {var_name} | {meta['asset']} | scale={scale} m")
        pieces = []
        for year in range(self.start_dt.year, self.end_dt.year + 1):
            stack, months = self._year_stack(meta, year)
            if stack is None: continue
            df = self._sample_stack(stack, months, var_name, scale)
            pieces.append(df)
        if not pieces: raise ValueError(f"[{var_name}] No months in range.")
        
        df = pd.concat(pieces, ignore_index=True).drop_duplicates(["station_id", "date"], keep="last")
        idx = pd.date_range(df["date"].min(), df["date"].max(), freq="MS")
        full = pd.MultiIndex.from_product([sorted(df["station_id"].unique()), idx], names=["station_id", "date"])
        return df.set_index(["station_id", "date"]).reindex(full).reset_index()

    def extract_static(self, asset_id, band_name, var_name="elev_m", reducer="mean", scale=None):
        if scale is None: scale = self.master_scale
        print(f"  → {var_name} (using {reducer} reducer at {scale}m scale)")
        if "MCD12Q1" in asset_id:
            img = ee.ImageCollection(asset_id).sort('system:time_start', False).first().select(band_name).rename(var_name)
        else:
            img = ee.Image(asset_id).select(band_name).rename(var_name)
        
        red_func = ee.Reducer.mode().setOutputs([var_name]) if reducer == "mode" else ee.Reducer.mean().setOutputs([var_name])
        features = img.reduceRegions(collection=self.gee_points, reducer=red_func, scale=scale).getInfo()["features"]
        
        rows = []
        for feat in features:
            props = feat.get("properties", {})
            sid = props.get("station_id")
            if not sid:
                xy = tuple(round(float(v), 5) for v in feat["geometry"]["coordinates"][:2])
                sid = self._xy_to_id.get(xy, "unknown")
            rows.append({"station_id": sid, var_name: props.get(var_name)})
        return pd.DataFrame(rows)

    def run_pipeline(self, spe_list, pet_product, dem_product, lulc_product, output_path):
        print(f"\n🚀 HydroSampler | {self.start_date} → {self.end_date}")
        frames = []
        for spe in spe_list:
            frames.append(self._extract_product(SPE_METADATA[spe], f"precip_{spe.lower()}"))
        frames.append(self._extract_product(PET_METADATA[pet_product], "pet"))

        merged = frames[0]
        for frame in frames[1:]:
            merged = merged.merge(frame, on=["station_id", "date"], how="outer", validate="one_to_one")

        dem = self.extract_static(DEM_METADATA[dem_product]["asset"], DEM_METADATA[dem_product]["band"], var_name="elev_m", reducer="mean", scale=DEM_METADATA[dem_product]["native_scale"])
        merged = merged.merge(dem, on="station_id", how="left")

        lulc = self.extract_static(LULC_METADATA[lulc_product]["asset"], LULC_METADATA[lulc_product]["band"], var_name="lulc_class", reducer="mode", scale=LULC_METADATA[lulc_product]["native_scale"])
        merged = merged.merge(lulc, on="station_id", how="left")
        merged = merged.merge(self.station_table, on="station_id", how="left", validate="many_to_one")

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(out, compression="snappy", index=False)
        print(f"✅ SUCCESS! Saved to: {out}")
        return merged


# ==============================================================================
# CMIP6 FUTURE PROJECTION EXTENSION (WITH HARGREAVES-SAMANI PET)
# ==============================================================================

CMIP6_METADATA = {
    "GDDP": {
        "asset": "NASA/GDDP-CMIP6",
        "bands": {
            "pr": {"unit": "kg_m-2_s-1", "conversion": "precip"},
            "tasmax": {"unit": "K", "conversion": "temp"},
            "tasmin": {"unit": "K", "conversion": "temp"}
        }
    }
}

class CMIP6Sampler(HydroSampler):
    def __init__(self, gauge_csv, col_station, col_lat, col_lon, 
                 start_date, end_date, project_id, 
                 scenario="ssp585", model="ACCESS-CM2", master_scale=27830):
        super().__init__(gauge_csv, col_station, col_lat, col_lon, 
                         start_date, end_date, project_id, 
                         master_scale=master_scale, sampling="native")
        self.scenario = scenario
        self.model = model
        print(f"✅ CMIP6 Sampler initialized: Scenario='{scenario}' | Model='{model}'")

    def _get_cmip6_collection(self, band_name):
        return (ee.ImageCollection(CMIP6_METADATA["GDDP"]["asset"])
                .filter(ee.Filter.eq('scenario', self.scenario))
                .filter(ee.Filter.eq('model', self.model))
                .filterDate(self.start_date, self.end_date)
                .select(band_name))

    def _cmip6_year_stack(self, band_name, conversion_type, year):
        ic = self._get_cmip6_collection(band_name)
        bands, months = [], []
        
        for month in range(1, 13):
            m_start, m_end = _month_bounds(year, month)
            if m_start < self.start_dt or m_start > self.end_dt: 
                continue
            
            monthly_ic = ic.filterDate(m_start, m_end)
            
            if conversion_type == "precip":
                monthly_img = monthly_ic.map(
                    lambda img: img.multiply(86400).copyProperties(img, ["system:time_start"])
                ).sum().rename(f"m{month:02d}")
            elif conversion_type == "temp":
                monthly_img = monthly_ic.map(
                    lambda img: img.subtract(273.15).copyProperties(img, ["system:time_start"])
                ).mean().rename(f"m{month:02d}")
                
            bands.append(monthly_img)
            months.append(pd.Timestamp(m_start))
            
        if not bands: 
            return None, []
        return ee.Image.cat(bands), months

    @staticmethod
    def _calculate_hargreaves_pet(row):
        tasmax = row['tasmax']
        tasmin = row['tasmin']
        lat_rad = np.radians(row['lat'])
        J = row['date'].dayofyear 
        
        Gsc = 0.0820
        dr = 1 + 0.033 * np.cos(2 * np.pi * J / 365)
        delta = 0.409 * np.sin(2 * np.pi * J / 365 - 1.39)
        ws = np.arccos(-np.tan(lat_rad) * np.tan(delta))
        
        Ra = (24 * 60 / np.pi) * Gsc * dr * (ws * np.sin(lat_rad) * np.sin(delta) + np.cos(lat_rad) * np.cos(delta) * np.sin(ws))
        
        temp_diff = np.clip(tasmax - tasmin, 0, None)
        tmean = (tasmax + tasmin) / 2.0
        
        pet_daily = 0.0023 * Ra * (tmean + 17.8) * np.sqrt(temp_diff)
        
        days_in_month = row['date'].days_in_month
        return pet_daily * days_in_month

    def extract_future_climate(self, output_path):
        print(f"\n🚀 CMIP6 Extraction | Scenario: {self.scenario} | Model: {self.model}")
        print("-" * 60)
        
        pieces_pr = []
        pieces_temp = []
        
        for year in range(self.start_dt.year, self.end_dt.year + 1):
            print(f"  -> Processing year {year}...")
            
            pr_stack, pr_months = self._cmip6_year_stack("pr", "precip", year)
            if pr_stack:
                pr_df = self._sample_stack(pr_stack, pr_months, "P_future", 27830)
                pieces_pr.append(pr_df)
                
            tasmax_stack, tasmax_months = self._cmip6_year_stack("tasmax", "temp", year)
            tasmin_stack, tasmin_months = self._cmip6_year_stack("tasmin", "temp", year)
            
            if tasmax_stack and tasmin_stack:
                tasmax_df = self._sample_stack(tasmax_stack, tasmax_months, "tasmax", 27830)
                tasmin_df = self._sample_stack(tasmin_stack, tasmin_months, "tasmin", 27830)
                
                temp_df = tasmax_df.merge(tasmin_df, on=["station_id", "date"], how="inner")
                temp_df["T_mean_future"] = (temp_df["tasmax"] + temp_df["tasmin"]) / 2.0
                pieces_temp.append(temp_df[["station_id", "date", "tasmax", "tasmin"]])
                
        if not pieces_pr:
            raise ValueError("No CMIP6 data extracted.")
            
        print("  -> Combining yearly data and calculating Hargreaves-Samani PET...")
        pr_final = pd.concat(pieces_pr, ignore_index=True)
        temp_final = pd.concat(pieces_temp, ignore_index=True) if pieces_temp else None
        
        merged = pr_final
        if temp_final is not None:
            merged = merged.merge(temp_final, on=["station_id", "date"], how="outer")
            
        merged = merged.merge(self.station_table, on="station_id", how="left")
        
        if 'tasmax' in merged.columns and 'tasmin' in merged.columns:
            merged['PET_future'] = merged.apply(self._calculate_hargreaves_pet, axis=1)
            merged = merged.drop(columns=['tasmax', 'tasmin'])
            
        merged['scenario'] = self.scenario
        merged['model'] = self.model
        
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(out, compression="snappy", index=False)
        
        print("-" * 60)
        print(f"✅ SUCCESS! Saved CMIP6 Future Data (with Hargreaves PET) to: {out}")
        print(f"   Shape: {merged.shape[0]} rows × {merged.shape[1]} columns")
        
        return merged