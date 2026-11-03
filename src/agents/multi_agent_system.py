import os
from typing import Dict, Any

class MonitoringAgent:
    """
    Observes incoming continuous streaming telemetry and detects threshold breaches.
    """
    def __init__(self, aqi_threshold=100.0):
        self.aqi_threshold = aqi_threshold

    def evaluate(self, telemetry: Dict[str, Any], predicted_aqi: float) -> Dict[str, Any]:
        is_breach = predicted_aqi > self.aqi_threshold
        dominant_pollutant = "PM2.5"
        if telemetry.get("pm10", 0) > telemetry.get("pm25", 0) * 1.8:
            dominant_pollutant = "PM10"
        elif telemetry.get("no2", 0) > 60:
            dominant_pollutant = "NO2"
        elif telemetry.get("o3", 0) > 80:
            dominant_pollutant = "O3"

        return {
            "status": "ANOMALY_TRIGGERED" if is_breach else "NORMAL",
            "predicted_aqi": predicted_aqi,
            "dominant_pollutant": dominant_pollutant,
            "severity": "CRITICAL" if predicted_aqi > 200 else ("ELEVATED" if is_breach else "NORMAL")
        }

class RecommendationAgent:
    """
    Formulates context-aware health advisories and municipal mitigation actions.
    """
    def generate_advisory(self, monitor_result: Dict[str, Any], category: str) -> Dict[str, Any]:
        aqi = monitor_result["predicted_aqi"]
        dominant = monitor_result["dominant_pollutant"]

        if aqi <= 50:
            public_health = "Air quality is considered satisfactory. Ambient air pollution poses little to no risk."
            actions = ["Maintain standard ventilation.", "Ideal conditions for outdoor physical activities."]
            municipal = "Normal municipal operations. Zero emergency restrictions needed."
        elif aqi <= 100:
            public_health = "Air quality is acceptable. However, unusually sensitive individuals should observe respiratory symptoms."
            actions = ["Sensitive groups should consider reducing prolonged outdoor exertion.", "Keep windows closed during peak traffic hours."]
            municipal = "Issue low-emission traffic route recommendations."
        elif aqi <= 150:
            public_health = f"Members of sensitive groups may experience health effects due to elevated {dominant}. General public is less likely to be affected."
            actions = ["Children, elderly, and individuals with respiratory conditions must limit outdoor exposure.", "Utilize N95 respirators if commuting outdoors.", "Operate indoor HEPA air purification systems."]
            municipal = "Activate dynamic speed limits on arterial corridors to curtail vehicular combustion emissions."
        elif aqi <= 200:
            public_health = f"Unhealthy environmental conditions. Increased likelihood of adverse effects and aggravation to heart and lungs across general public."
            actions = ["Avoid all strenuous outdoor physical exertion.", "Seal building fresh-air intakes and run air conditioning on recirculate mode.", "Wear protective masks when traversing outdoor urban zones."]
            municipal = "Enforce temporary industrial smoke scrubbers and restrict heavy diesel freight vehicles."
        else:
            public_health = f"EMERGENCY HEALTH WARNING: Severe environmental hazard driven by acute {dominant} concentration."
            actions = ["Remain strictly indoors.", "Run high-capacity particulate filters continuously.", "Seek medical evaluation if experiencing shortness of breath."]
            municipal = "Deploy municipal misting cannons, halt unpaved construction activities, and divert urban traffic."

        return {
            "category": category,
            "public_health_guideline": public_health,
            "recommended_actions": actions,
            "municipal_response": municipal
        }

class ReportingAgent:
    """
    Handles interactive inquiries, explaining environmental telemetry and reasoning.
    """
    def respond_to_query(self, query: str, current_state: Dict[str, Any]) -> str:
        q = query.lower()
        aqi = current_state.get("aqi", 45)
        cat = current_state.get("category", "Good")
        dom = current_state.get("dominant_pollutant", "PM2.5")

        if "safe" in q or "exercise" in q or "walk" in q:
            if aqi <= 100:
                return f"Yes, the current Air Quality Index is {aqi:.1f} ({cat}). Outdoor workouts are safe for most individuals."
            else:
                return f"Caution advised. With an AQI of {aqi:.1f} ({cat}), outdoor workouts are not recommended, particularly due to high {dom} levels."
        elif "mask" in q or "n95" in q:
            if aqi > 100:
                return f"Yes, wearing an N95 respirator is strongly advised given the current AQI of {aqi:.1f} ({cat})."
            else:
                return f"A mask is not necessary under current good air quality levels (AQI {aqi:.1f})."
        elif "pollutant" in q or "cause" in q or "source" in q:
            return f"The dominant contributing pollutant is currently {dom}, driven by particulate matter and ambient combustion."
        else:
            return f"Current environmental assessment: AQI is {aqi:.1f} ({cat}). The multi-agent system is continuously monitoring telemetry for real-time safety."

class MultiAgentOrchestrator:
    """
    Coordinates the LangChain-style prompt chaining and agent workflow.
    """
    def __init__(self, predictor):
        self.predictor = predictor
        self.monitoring_agent = MonitoringAgent()
        self.recommendation_agent = RecommendationAgent()
        self.reporting_agent = ReportingAgent()

    def process_telemetry(self, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        predicted_aqi = self.predictor.predict(telemetry)
        cat_info = self.predictor.get_aqi_category(predicted_aqi)

        mon_eval = self.monitoring_agent.evaluate(telemetry, predicted_aqi)
        advisory = self.recommendation_agent.generate_advisory(mon_eval, cat_info["category"])

        result = {
            "telemetry": telemetry,
            "predicted_aqi": predicted_aqi,
            "category": cat_info["category"],
            "severity": cat_info["severity"],
            "color": cat_info["color"],
            "monitoring_status": mon_eval["status"],
            "dominant_pollutant": mon_eval["dominant_pollutant"],
            "advisory": advisory
        }
        return result
