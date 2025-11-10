"""Medical domain scenario - Full patient case study (Phase 2)."""

from typing import Dict, List, Set

from .base import Scenario


class MedicalLongScenario(Scenario):
    """Complete patient case study compression task - realistic length."""

    @property
    def name(self) -> str:
        return "medical_long"

    @property
    def domain(self) -> str:
        return "medical"

    @property
    def original(self) -> str:
        return (
            "A 67-year-old male patient with history of hypertension and hyperlipidemia presented "
            "to the emergency department on March 15, 2024 at 08:45 AM with acute onset chest pain "
            "that began approximately 2 hours prior. The pain was described as severe pressure-like "
            "discomfort, rated 8/10 in intensity, radiating to the left arm and jaw. Associated "
            "symptoms included diaphoresis, nausea, and shortness of breath. The patient denied "
            "previous similar episodes. Initial vital signs revealed blood pressure 165/95 mmHg, "
            "heart rate 108 beats per minute, respiratory rate 22 breaths per minute, temperature "
            "98.6°F, and oxygen saturation 94% on room air. Physical examination revealed an anxious "
            "male in moderate distress with clear lung sounds bilaterally and regular heart rhythm "
            "without murmurs. Laboratory analysis showed troponin-I elevated at 2.8 ng/mL (normal "
            "range less than 0.04 ng/mL), creatine kinase-MB 45 U/L, BNP 380 pg/mL, and creatinine "
            "1.1 mg/dL. Complete blood count was within normal limits. Initial 12-lead electrocardiogram "
            "revealed ST-segment elevation of 3 mm in leads II, III, and aVF, consistent with acute "
            "inferior wall ST-elevation myocardial infarction. The patient was immediately given "
            "aspirin 325 mg, clopidogrel 600 mg loading dose, atorvastatin 80 mg, and intravenous "
            "heparin bolus followed by continuous infusion. He was emergently transferred to the "
            "cardiac catheterization laboratory at 10:15 AM. Coronary angiography identified 95% "
            "stenosis of the proximal right coronary artery with TIMI grade 1 flow and thrombus "
            "formation. Percutaneous coronary intervention was performed with successful deployment "
            "of a drug-eluting stent measuring 3.5 mm in diameter by 18 mm in length. Final "
            "angiography showed TIMI grade 3 flow with 0% residual stenosis. Post-procedure "
            "troponin-I peaked at 8.2 ng/mL at 6 hours. The patient was transferred to the coronary "
            "care unit in stable condition. Hospital course was uncomplicated. Transthoracic "
            "echocardiography on day 2 showed left ventricular ejection fraction 48% with mild "
            "hypokinesis of the inferior wall. Patient was discharged on day 3 on optimal medical "
            "therapy including aspirin 81 mg daily, clopidogrel 75 mg daily for 12 months, "
            "atorvastatin 80 mg nightly, metoprolol succinate 50 mg twice daily, and lisinopril "
            "10 mg daily. Exercise stress test at 6 weeks post-discharge showed good functional "
            "capacity with no inducible ischemia or arrhythmias. At 3-month follow-up, patient "
            "remains asymptomatic with well-controlled blood pressure and lipids."
        )

    @property
    def facts(self) -> List[str]:
        return [
            "67-year-old male",
            "hypertension and hyperlipidemia history",
            "presented March 15, 2024 at 08:45 AM",
            "chest pain onset 2 hours prior",
            "pain severity 8/10",
            "radiating to left arm and jaw",
            "blood pressure 165/95 mmHg",
            "heart rate 108 bpm",
            "respiratory rate 22 breaths per minute",
            "oxygen saturation 94%",
            "troponin-I 2.8 ng/mL",
            "creatine kinase-MB 45 U/L",
            "BNP 380 pg/mL",
            "ST-elevation 3 mm in leads II, III, aVF",
            "inferior wall STEMI",
            "aspirin 325 mg given",
            "clopidogrel 600 mg loading dose",
            "atorvastatin 80 mg",
            "transferred to cath lab 10:15 AM",
            "95% stenosis of proximal RCA",
            "drug-eluting stent 3.5 mm x 18 mm deployed",
            "post-procedure troponin peaked 8.2 ng/mL at 6 hours",
            "ejection fraction 48%",
            "discharged day 3",
            "aspirin 81 mg daily",
            "clopidogrel 75 mg daily for 12 months",
            "metoprolol 50 mg twice daily",
            "lisinopril 10 mg daily",
            "stress test at 6 weeks showed no ischemia",
            "asymptomatic at 3-month follow-up",
        ]

    @property
    def banned(self) -> Set[str]:
        return {"died", "expired", "fatal", "deceased", "unsuccessful"}

    @property
    def alias_map(self) -> Dict[str, List[str]]:
        return {
            "67-year-old male": ["67yo M", "67M", "male 67"],
            "hypertension and hyperlipidemia history": ["HTN/HLD hx", "h/o HTN, HLD"],
            "presented March 15, 2024 at 08:45 AM": ["3/15/24 0845", "presented 3/15/24 08:45"],
            "chest pain onset 2 hours prior": ["CP onset 2h", "chest pain 2h prior"],
            "pain severity 8/10": ["8/10 pain", "pain 8/10"],
            "radiating to left arm and jaw": ["radiating L arm/jaw", "L arm/jaw radiation"],
            "blood pressure 165/95 mmHg": ["BP 165/95", "165/95"],
            "heart rate 108 bpm": ["HR 108", "108 bpm"],
            "respiratory rate 22 breaths per minute": ["RR 22", "22 breaths/min"],
            "oxygen saturation 94%": ["O2 sat 94%", "SpO2 94%"],
            "troponin-I 2.8 ng/mL": ["trop-I 2.8", "troponin 2.8"],
            "creatine kinase-MB 45 U/L": ["CK-MB 45", "CKMB 45"],
            "BNP 380 pg/mL": ["BNP 380"],
            "ST-elevation 3 mm in leads II, III, aVF": ["STE 3mm II/III/aVF", "3mm STE inferior leads"],
            "inferior wall STEMI": ["inferior STEMI", "IWMI"],
            "aspirin 325 mg given": ["ASA 325mg", "aspirin 325"],
            "clopidogrel 600 mg loading dose": ["Plavix 600mg load", "clopidogrel 600"],
            "atorvastatin 80 mg": ["atorva 80mg", "statin 80"],
            "transferred to cath lab 10:15 AM": ["to cath lab 1015", "cath lab 10:15"],
            "95% stenosis of proximal RCA": ["95% prox RCA stenosis", "RCA 95% stenosis"],
            "drug-eluting stent 3.5 mm x 18 mm deployed": ["DES 3.5x18mm", "stent 3.5x18"],
            "post-procedure troponin peaked 8.2 ng/mL at 6 hours": ["peak trop 8.2 at 6h", "troponin peak 8.2"],
            "ejection fraction 48%": ["EF 48%", "LVEF 48%"],
            "discharged day 3": ["d/c day 3", "discharge day 3"],
            "aspirin 81 mg daily": ["ASA 81mg qd", "aspirin 81 daily"],
            "clopidogrel 75 mg daily for 12 months": ["Plavix 75mg x12mo", "clopidogrel 75 x 12 months"],
            "metoprolol 50 mg twice daily": ["metoprolol 50mg BID", "metop 50 BID"],
            "lisinopril 10 mg daily": ["lisinopril 10mg qd", "lisinopril 10 daily"],
            "stress test at 6 weeks showed no ischemia": ["6wk stress test negative", "no ischemia at 6 weeks"],
            "asymptomatic at 3-month follow-up": ["asymptomatic 3mo f/u", "3mo f/u asymptomatic"],
        }

    @property
    def difficulty(self) -> str:
        return "hard"
