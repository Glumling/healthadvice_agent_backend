# agent_backend/tools.py
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import requests, re, random, json, time, math, pandas as pd
from pathlib import Path

# ───────────────────────────────────────────
# Local offline exercise dataframe loader
# ───────────────────────────────────────────
_DATA = Path("data/exercises.csv")          # Kaggle dataset (≈1 MB) :contentReference[oaicite:2]{index=2}
_DF   = None
def _ex_df():
    global _DF
    if _DF is None:
        if _DATA.exists():
            _DF = pd.read_csv(_DATA)
            _DF["name_low"] = _DF["name"].str.lower()
            _DF["target_low"] = _DF["target"].str.lower()
        else:
            _DF = pd.DataFrame(columns=["name", "target", "name_low", "target_low"])
    return _DF

# ═══════════════════════════════════════════
#  1. Body-metrics & nutrition tools
# ═══════════════════════════════════════════
class BMIInput(BaseModel):
    weight: float = Field(..., description="kg")
    height: float = Field(..., description="cm")

@tool(args_schema=BMIInput)
def calculate_bmi(weight: float, height: float) -> str:
    """Compute Body-Mass Index (kg/m²)."""
    bmi = weight / (height/100)**2
    return f"Your BMI is {bmi:.1f}."

class CalorieIn(BaseModel):
    weight: float; height: float; age: int; gender: str
    activity: float = Field(..., description="Activity factor 1.2–1.9")

@tool(args_schema=CalorieIn)
def estimate_calories(**kw) -> str:
    """Mifflin-St Jeor BMR × activity factor :contentReference[oaicite:3]{index=3}."""
    w,h,a,g,act = kw.values()
    bmr = 10*w + 6.25*h - 5*a + (5 if g.lower().startswith("m") else -161)
    return f"Maintenance ≈ {bmr*act:.0f} kcal/d."

class Age(BaseModel): age:int
@tool(args_schema=Age)
def target_hr(age:int)->str:
    """50–85 % of theoretical max (220-age)."""
    max_hr = 220-age
    return f"Target zone {int(.5*max_hr)}-{int(.85*max_hr)} bpm."

# ═══════ Units / hydration / macros / split ═══════
class Convert(BaseModel):
    value: float; unit:str = Field(...,description="kg,lb,cm,in")
@tool(args_schema=Convert)
def unit_convert(value:float, unit:str)->str:
    """kg↔lb & cm↔in converter."""
    u=unit.lower()
    if u=="kg":return f"{value:.1f} kg ≈ {value*2.20462:.1f} lb"
    if u=="lb":return f"{value:.1f} lb ≈ {value/2.20462:.1f} kg"
    if u=="cm":return f"{value:.1f} cm ≈ {value/2.54:.1f} in"
    if u=="in":return f"{value:.1f} in ≈ {value*2.54:.1f} cm"
    return "Unit must be kg, lb, cm or in."

class Water(BaseModel): weight_kg:float
@tool(args_schema=Water)
def water_goal(weight_kg:float)->str:
    """35 ml/kg daily hydration guideline :contentReference[oaicite:4]{index=4}."""
    ml=weight_kg*35
    return f"Target ≈ {ml/1000:.2f} L ({ml:.0f} ml)."

class Macro(BaseModel):
    calories:int; split:str="40/30/30"
@tool(args_schema=Macro)
def macro_split(calories:int, split:str)->str:
    """Grams carbs/protein/fat for kcal & % split."""
    try:c,p,f=map(float,split.split("/"))
    except: return "Split like 40/30/30."
    if abs(c+p+f-100)>0.1: return "Percents must sum to 100."
    return (f"{calories*c/4/100:.0f} g carbs, "
            f"{calories*p/4/100:.0f} g protein, "
            f"{calories*f/9/100:.0f} g fat.")

@tool
def workout_split()->str:
    """Balanced 7-day push/pull/legs/full programme."""
    return ("\n".join([
      "Mon Push", "Tue Pull", "Wed Legs", "Thu Rest/Mobility",
      "Fri Upper Hypertrophy", "Sat Lower Hypertrophy",
      "Sun Active Recovery"
    ]))

# ═══════ Exercise database tools (offline CSV) ═══════
class Query(BaseModel): query:str
@tool(args_schema=Query)
def free_db_search(query:str)->str:
    """Find exercises whose name contains a keyword."""
    df=_ex_df()
    hits=df[df["name_low"].str.contains(re.escape(query.lower()))]
    return "\n".join(hits["name"].head(5)) or f"No matches for {query}."

class Muscle(BaseModel): muscle:str
@tool(args_schema=Muscle)
def exercises_by_muscle(muscle:str)->str:
    """Up to 5 exercises targeting a muscle group."""
    df=_ex_df()
    hits=df[df["target_low"].str.contains(re.escape(muscle.lower()))]
    if hits.empty: return f"No exercises for {muscle}."
    return "\n".join(random.sample(list(hits["name"]), k=min(5,len(hits))))

# ═══════ Meal & barcode look-ups (free APIs) ═══════
class Ingredient(BaseModel): ingredient:str
@tool(args_schema=Ingredient)
def recipes_by_ingredient(ingredient:str)->str:
    """List up to 5 meals containing an ingredient (TheMealDB)."""
    meals=requests.get(
        f"https://www.themealdb.com/api/json/v1/1/filter.php?i={ingredient}"
    ).json().get("meals") or []            # :contentReference[oaicite:5]{index=5}
    return ", ".join(m["strMeal"] for m in meals[:5]) or "No recipes."

class Barcode(BaseModel): barcode:str
@tool(args_schema=Barcode)
def product_by_barcode(barcode:str)->str:
    """Open Food Facts product + nutriscore :contentReference[oaicite:6]{index=6}."""
    data=requests.get(
        f"https://world.openfoodfacts.org/api/v0/product/{barcode}.json"
    ).json()
    if data.get("status")!=1: return "Product not found."
    p=data["product"]
    return f"{p.get('product_name','Unknown')} – Nutriscore {p.get('nutriscore_grade','?').upper()}"

# ═══════ Strength & cardio calculators ═══════
class ORM(BaseModel): weight:float; reps:int
@tool(args_schema=ORM)
def one_rep_max(weight:float, reps:int)->str:
    """Epley 1-RM estimate :contentReference[oaicite:7]{index=7}."""
    return f"1-RM ≈ {(weight*(1+reps/30)):.1f} kg"

class VO2In(BaseModel): age:int; gender:str; resting_hr:int
@tool(args_schema=VO2In)
def vo2max(age:int, gender:str, resting_hr:int)->str:
    """VO₂ max from HRmax/HRrest ratio (Uth–Sorensen) :contentReference[oaicite:8]{index=8}."""
    return f"VO₂ max ≈ {15.3*(220-age)/resting_hr:.1f} ml/kg/min"

@tool
def rpe_table()->str:
    """RPE 6-10 to %1-RM quick-ref."""
    return "\n".join(f"RPE {r} ≈ {100-(r-6)*5}% 1-RM" for r in range(6,11))

class HIIT(BaseModel):
    work:int=Field(...,description="sec"); rest:int; rounds:int
@tool(args_schema=HIIT)
def hiit_plan(work:int, rest:int, rounds:int)->str:
    """Summarise a HIIT block total time."""
    total=(work+rest)*rounds; return f"{rounds}×{work}/{rest}s → {total//60}:{total%60:02d} min"

@tool
def stretch_routine()->str:
    """5-move full-body stretch sequence."""
    return "\n".join([
      "Cat-Cow ×10","World’s Greatest Stretch ×5/side",
      "Hip Flexor 30 s/side","Hamstring Doorway 30 s/side",
      "Shoulder Pass-throughs ×15"])

class Sleep(BaseModel): goal_hours:float=8; actual_hours:float
@tool(args_schema=Sleep)
def sleep_debt(goal_hours:float, actual_hours:float)->str:
    """Report daily sleep debt vs. goal."""
    debt=goal_hours-actual_hours
    return ("Goal met! 🎉" if debt<=0 else f"Sleep debt: {debt:.1f} h")
