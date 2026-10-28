"""
MedQuery: Structured Medical Knowledge Base
Provides clinical descriptions, urgency triage levels, precautions, specialist referrals,
and graph-based relation retrieval for all 41 diagnostic conditions.
"""

from typing import Dict, List, Optional


# Structured Clinical Knowledge Base for all 41 conditions
DISEASE_KNOWLEDGE: Dict[str, Dict] = {
    "Fungal Infection": {
        "description": "A superficial mycotic skin infection caused by dermatophytes or yeast, resulting in localized inflammation, scaling, and pruritus.",
        "urgency": "Routine",
        "specialist": "Dermatologist",
        "precautions": [
            "Keep the affected skin clean, dry, and exposed to air when possible.",
            "Avoid sharing towels, clothing, or footwear.",
            "Use antifungal cream or powder as prescribed by a doctor.",
            "Wear loose, breathable cotton clothing."
        ],
        "warning_signs": ["Spreading erythema with warmth and pus", "Fever", "Severe pain"]
    },
    "Allergy": {
        "description": "An immunological hypersensitivity reaction triggered by allergens such as pollen, pet dander, foods, or dust mites.",
        "urgency": "Routine",
        "specialist": "Allergist / Immunologist",
        "precautions": [
            "Identify and avoid known environmental or dietary triggers.",
            "Take prescribed antihistamines if symptoms flare.",
            "Keep living areas free of dust and allergen reservoirs.",
            "Consult a physician if facial swelling or breathing difficulty occurs."
        ],
        "warning_signs": ["Difficulty breathing or wheezing", "Swelling of lips, tongue, or throat (Anaphylaxis)", "Dizziness"]
    },
    "Gastroesophageal Reflux Disease (GERD)": {
        "description": "A digestive condition where stomach acid or bile frequently flows back into the esophagus, irritating the mucosal lining.",
        "urgency": "Routine",
        "specialist": "Gastroenterologist",
        "precautions": [
            "Eat smaller, more frequent meals rather than heavy dinners.",
            "Avoid lying down for at least 2-3 hours after eating.",
            "Limit spicy, acidic, fatty foods, caffeine, and alcohol.",
            "Elevate the head of your bed by 6 inches."
        ],
        "warning_signs": ["Difficulty swallowing (dysphagia)", "Unexplained weight loss", "Persistent vomiting or vomiting blood"]
    },
    "Chronic Cholestasis": {
        "description": "A progressive hepatic disorder characterized by impaired bile formation or excretion leading to retention of biliary components.",
        "urgency": "Moderate",
        "specialist": "Hepatologist / Gastroenterologist",
        "precautions": [
            "Consult a hepatologist promptly for comprehensive liver function panel.",
            "Avoid all alcohol and hepatotoxic medications.",
            "Maintain adequate fat-soluble vitamin (A, D, E, K) supplementation under guidance.",
            "Report any darkening of urine or pale stools."
        ],
        "warning_signs": ["Severe jaundice with confusion", "Ascites (abdominal swelling)", "Bleeding tendency"]
    },
    "Drug Reaction": {
        "description": "An adverse immune or non-immune response of the body following the ingestion or injection of a pharmaceutical agent.",
        "urgency": "Moderate",
        "specialist": "General Physician / Allergist",
        "precautions": [
            "Immediately stop suspected offending non-essential medications under doctor supervision.",
            "Document the medication name, dose, and time of onset.",
            "Stay well hydrated and avoid applying harsh soaps to rashes.",
            "Seek emergency care if skin blisters or mucosal surfaces (mouth, eyes) are involved."
        ],
        "warning_signs": ["Skin peeling, mucosal blisters (Stevens-Johnson syndrome)", "High fever with generalized rash", "Difficulty breathing"]
    },
    "Peptic Ulcer Disease": {
        "description": "Erosions or sores in the mucosal lining of the stomach or duodenum, commonly caused by H. pylori infection or prolonged NSAID use.",
        "urgency": "Moderate",
        "specialist": "Gastroenterologist",
        "precautions": [
            "Avoid nonsteroidal anti-inflammatory drugs (NSAIDs) such as ibuprofen and aspirin.",
            "Refrain from smoking and consuming alcohol or heavy caffeine.",
            "Eat bland, non-irritating foods at regular intervals.",
            "Complete antibiotic and proton-pump inhibitor courses if prescribed."
        ],
        "warning_signs": ["Black tarry stools (melena)", "Vomiting blood or coffee-ground emesis", "Sudden sharp agonizing abdominal pain"]
    },
    "AIDS / HIV": {
        "description": "An advanced stage of Human Immunodeficiency Virus infection that damages immune cells, predisposing to opportunistic infections.",
        "urgency": "High",
        "specialist": "Infectious Disease Specialist",
        "precautions": [
            "Adhere strictly to prescribed Antiretroviral Therapy (ART) regimen.",
            "Maintain rigorous personal hygiene and food safety standards.",
            "Receive recommended non-live immunizations under specialist care.",
            "Schedule regular CD4 cell count and viral load monitoring."
        ],
        "warning_signs": ["Persistent high fevers and night sweats", "Unexplained profound weight loss", "Shortness of breath or persistent cough"]
    },
    "Diabetes": {
        "description": "A metabolic disorder characterized by chronic hyperglycemia resulting from defects in insulin secretion, action, or both.",
        "urgency": "Moderate",
        "specialist": "Endocrinologist",
        "precautions": [
            "Monitor blood glucose levels regularly as instructed by your clinician.",
            "Follow a balanced low-glycemic, fiber-rich nutritional plan.",
            "Engage in regular moderate aerobic physical activity.",
            "Perform daily foot examinations to detect any skin breakdown early."
        ],
        "warning_signs": ["Extreme confusion, fruity breath odor, rapid breathing (DKA)", "Severe hypoglycemia (<70 mg/dL)", "Non-healing ulcers"]
    },
    "Gastroenteritis": {
        "description": "Inflammation of the gastrointestinal tract, often termed 'stomach flu', caused by viral or bacterial infection.",
        "urgency": "Routine",
        "specialist": "General Physician",
        "precautions": [
            "Maintain vigorous hydration with Oral Rehydration Salts (ORS) or electrolyte solutions.",
            "Adopt the BRAT diet (bananas, rice, applesauce, toast) until stools normalize.",
            "Wash hands thoroughly after using the bathroom and before meals.",
            "Avoid dairy, greasy foods, and sugary beverages during recovery."
        ],
        "warning_signs": ["Inability to keep liquids down for >24 hours", "Signs of severe dehydration (sunken eyes, no urine)", "High fever or bloody stool"]
    },
    "Bronchial Asthma": {
        "description": "A chronic inflammatory airway condition characterized by bronchial hyperreactivity, episodic wheezing, and breathlessness.",
        "urgency": "Moderate",
        "specialist": "Pulmonologist",
        "precautions": [
            "Always keep your fast-acting rescue inhaler (bronchodilator) accessible.",
            "Avoid known triggers such as tobacco smoke, cold air, pet fur, and pollen.",
            "Monitor peak expiratory flow rate if recommended by your physician.",
            "Follow your personalized asthma action plan."
        ],
        "warning_signs": ["Severe breathlessness speaking only in broken words", "Cyanosis (bluish lips)", "No relief from rescue inhaler"]
    },
    "Hypertension": {
        "description": "Persistent elevation of systemic arterial blood pressure, which increases cardiovascular and renal morbidity if untreated.",
        "urgency": "Moderate",
        "specialist": "Cardiologist",
        "precautions": [
            "Reduce dietary sodium intake (<2,000 mg/day) and follow the DASH diet.",
            "Exercise moderately for at least 30 minutes on most days of the week.",
            "Take prescribed antihypertensive drugs consistently without self-stopping.",
            "Keep a daily or weekly home blood pressure log."
        ],
        "warning_signs": ["Blood pressure >180/120 mmHg (Hypertensive Crisis)", "Severe headache with vision blurring", "Chest pain or shortness of breath"]
    },
    "Migraine": {
        "description": "A recurrent neurovascular headache disorder characterized by moderate-to-severe throbbing unilateral head pain, nausea, and sensory sensitivity.",
        "urgency": "Routine",
        "specialist": "Neurologist",
        "precautions": [
            "Rest in a quiet, darkened room during acute episodes.",
            "Apply a cool compress or ice pack to the forehead or temples.",
            "Identify and track triggers (stress, irregular sleep, certain aged foods, flashing lights).",
            "Take prescribed triptans or analgesics at the earliest onset of pain."
        ],
        "warning_signs": ["Sudden explosive 'thunderclap' headache", "Headache accompanied by fever and stiff neck", "New persistent focal neurological deficit"]
    },
    "Cervical Spondylosis": {
        "description": "Age-related degenerative wear and tear affecting the cervical spinal vertebrae and intervertebral discs.",
        "urgency": "Routine",
        "specialist": "Orthopedic Surgeon / Physiotherapist",
        "precautions": [
            "Maintain ergonomic posture while working on computers or mobile devices.",
            "Perform gentle neck strengthening and range-of-motion exercises.",
            "Use a supportive ergonomic pillow that keeps the neck aligned.",
            "Avoid lifting heavy weights overhead."
        ],
        "warning_signs": ["Shooting radiating pain or numbness down both arms", "Loss of hand dexterity or grip strength", "Gait unsteadiness"]
    },
    "Paralysis (Brain Hemorrhage)": {
        "description": "A critical neurovascular emergency where an intracerebral vessel ruptures, causing localized hemorrhage and neurological deficit.",
        "urgency": "Emergency",
        "specialist": "Neurologist / Emergency Medicine",
        "precautions": [
            "CALL EMERGENCY MEDICAL SERVICES (911/112) IMMEDIATELY.",
            "Keep the patient calm, resting with the head slightly elevated.",
            "Do NOT give anything to eat, drink, or medications like aspirin.",
            "Note the exact time symptoms started (critical for thrombolysis/intervention window)."
        ],
        "warning_signs": ["Facial drooping", "Arm weakness on one side", "Slurred speech (FAST criteria)"]
    },
    "Jaundice": {
        "description": "A yellow discoloration of the skin, sclera, and mucous membranes caused by hyperbilirubinemia due to liver or biliary pathology.",
        "urgency": "Moderate",
        "specialist": "Gastroenterologist / Hepatologist",
        "precautions": [
            "Consult a doctor for serum bilirubin, ALT, AST, and abdominal ultrasound.",
            "Consume easily digestible, low-fat meals and drink plenty of fluids.",
            "Completely abstain from alcohol consumption.",
            "Avoid self-medicating with over-the-counter painkillers."
        ],
        "warning_signs": ["Extreme lethargy or confusion (encephalopathy)", "Vomiting blood", "High fever with chills"]
    },
    "Malaria": {
        "description": "A life-threatening mosquito-borne infectious disease caused by Plasmodium parasites, characterized by paroxysms of chills and high fever.",
        "urgency": "High",
        "specialist": "Infectious Disease Specialist",
        "precautions": [
            "Undergo immediate blood smear microscopy / rapid antigen testing.",
            "Complete full course of artemisinin-based combination therapies as prescribed.",
            "Use insecticide-treated bed nets and insect repellents.",
            "Maintain adequate hydration and take antipyretics for fever control."
        ],
        "warning_signs": ["Confusion or seizures (Cerebral Malaria)", "Severe anemia with pale conjunctiva", "Difficulty breathing"]
    },
    "Chicken Pox": {
        "description": "A highly contagious viral infection caused by Varicella-Zoster Virus, presenting with generalized itchy blister-like rashes and fever.",
        "urgency": "Routine",
        "specialist": "General Physician / Pediatrician",
        "precautions": [
            "Isolate the patient to prevent transmission until all blisters have crusted over.",
            "Apply calamine lotion and take cool oatmeal baths to relieve intense pruritus.",
            "Keep fingernails short and clean to prevent bacterial superinfection from scratching.",
            "Never give aspirin to children with chickenpox (risk of Reye's syndrome)."
        ],
        "warning_signs": ["Rash spreading to eyes", "Skin lesions becoming warm, extremely red, or swollen with pus", "High fever or cough"]
    },
    "Dengue": {
        "description": "A mosquito-borne viral disease caused by Dengue virus, manifesting with high fever, severe retro-orbital headache, and muscle/joint pains.",
        "urgency": "High",
        "specialist": "General Physician / Infectious Disease Specialist",
        "precautions": [
            "Obtain daily platelet count and hematocrit monitoring.",
            "Avoid NSAIDs (ibuprofen, aspirin) due to bleeding risks; use only paracetamol for fever.",
            "Maintain aggressive oral fluid intake (coconut water, ORS, fresh juices).",
            "Rest completely and use mosquito netting to prevent transmission."
        ],
        "warning_signs": ["Severe abdominal pain", "Persistent vomiting", "Bleeding from gums or nose", "Extreme fatigue or restlessness"]
    },
    "Typhoid": {
        "description": "A systemic bacterial infection caused by Salmonella enterica serotype Typhi, presenting with step-ladder fever and toxic appearance.",
        "urgency": "High",
        "specialist": "Infectious Disease Specialist",
        "precautions": [
            "Complete the entire course of prescribed antibiotics (do not discontinue early).",
            "Drink only boiled, filtered, or commercially bottled water.",
            "Eat freshly cooked, hot, soft foods; avoid raw vegetables or street food.",
            "Maintain strict hand washing protocols."
        ],
        "warning_signs": ["Severe abdominal distension or acute belly pain (intestinal perforation)", "Confusion or delirium", "High persistent fever"]
    },
    "Hepatitis A": {
        "description": "An acute viral liver infection transmitted primarily via the fecal-oral route through contaminated food or water.",
        "urgency": "Moderate",
        "specialist": "Gastroenterologist",
        "precautions": [
            "Rest adequately and allow the body time to recover.",
            "Consume a high-calorie, balanced, low-fat diet.",
            "Strictly avoid alcohol until liver enzymes completely normalize.",
            "Practice meticulous hand hygiene to protect family members."
        ],
        "warning_signs": ["Severe vomiting unable to retain fluids", "Confusion or altered mental state", "Dark urine with clay-colored stools"]
    },
    "Hepatitis B": {
        "description": "A viral infection that attacks the liver and can cause both acute and chronic disease, transmitted via blood and bodily fluids.",
        "urgency": "High",
        "specialist": "Hepatologist",
        "precautions": [
            "Follow up with a hepatologist for viral DNA load and liver ultrasound.",
            "Ensure household members and sexual partners receive the Hepatitis B vaccine.",
            "Never share razors, needles, or toothbrushes.",
            "Avoid alcohol and consult before taking any new supplement."
        ],
        "warning_signs": ["Fluid accumulation in abdomen (ascites)", "Confusion or drowsiness", "Upper gastrointestinal bleeding"]
    },
    "Hepatitis C": {
        "description": "A blood-borne viral infection causing chronic liver inflammation, frequently asymptomatic until advanced liver disease develops.",
        "urgency": "High",
        "specialist": "Hepatologist",
        "precautions": [
            "Seek evaluation for curative direct-acting antiviral (DAA) therapy.",
            "Avoid all alcohol and hepatotoxic substances.",
            "Screen for liver fibrosis or cirrhosis periodically.",
            "Do not donate blood or organs."
        ],
        "warning_signs": ["Jaundice", "Spider angiomas", "Significant unexplained abdominal swelling"]
    },
    "Hepatitis D": {
        "description": "A defective RNA virus that requires Hepatitis B virus envelope proteins for replication, often resulting in severe progressive liver injury.",
        "urgency": "High",
        "specialist": "Hepatologist",
        "precautions": [
            "Coordinate close management with specialized tertiary hepatology center.",
            "Adhere to antiviral and interferon therapy protocols as directed.",
            "Maintain total abstinence from alcohol.",
            "Monitor liver function and alpha-fetoprotein routinely."
        ],
        "warning_signs": ["Rapid onset jaundice", "Bleeding diathesis", "Mental disorientation"]
    },
    "Hepatitis E": {
        "description": "An enterically transmitted viral infection of the liver, particularly dangerous with high mortality in pregnant individuals.",
        "urgency": "High",
        "specialist": "Gastroenterologist / Obstetrician",
        "precautions": [
            "Drink only safe boiled or treated water.",
            "Ensure thoroughly cooked food, particularly meat products.",
            "Immediate hospital monitoring if patient is pregnant.",
            "Rest and hydration during the acute convalescent phase."
        ],
        "warning_signs": ["Pregnancy with jaundice (obstetric emergency)", "Severe coagulopathy", "Lethargy and vomiting"]
    },
    "Alcoholic Hepatitis": {
        "description": "Acute or chronic liver inflammation and necrosis resulting from heavy and prolonged alcohol consumption.",
        "urgency": "High",
        "specialist": "Hepatologist / Addiction Specialist",
        "precautions": [
            "Total, permanent cessation of alcohol intake is mandatory for survival.",
            "Seek inpatient medical detoxification to manage withdrawal safely.",
            "Work with a clinical nutritionist for high-protein nutritional support.",
            "Take prescribed corticosteroids or pentoxifylline if indicated by severity scoring."
        ],
        "warning_signs": ["Abdominal fluid distention (ascites)", "Disorientation or asterixis", "Hematemesis (vomiting blood)"]
    },
    "Tuberculosis": {
        "description": "A contagious granulomatous infection caused by Mycobacterium tuberculosis, primarily attacking the lungs.",
        "urgency": "High",
        "specialist": "Pulmonologist / Infectious Disease Specialist",
        "precautions": [
            "Adhere rigorously to Directly Observed Treatment Short-course (DOTS) antibiotic therapy.",
            "Wear a medical mask in enclosed spaces during the infectious phase.",
            "Maintain good room ventilation and sleep in a separate well-ventilated room.",
            "Eat nutrient-dense, high-protein meals to aid tissue repair."
        ],
        "warning_signs": ["Coughing up frank blood (hemoptysis)", "Sudden chest pain with severe shortness of breath", "Severe unremitting fever"]
    },
    "Common Cold": {
        "description": "A mild, self-limiting viral upper respiratory tract infection predominantly caused by rhinoviruses.",
        "urgency": "Routine",
        "specialist": "General Physician",
        "precautions": [
            "Get plenty of rest and drink warm liquids (tea, broth, warm water).",
            "Use saline nasal sprays or steam inhalation to alleviate nasal congestion.",
            "Wash hands frequently and cover coughs and sneezes.",
            "Over-the-counter decongestants may provide symptomatic relief."
        ],
        "warning_signs": ["Fever lasting >4 days", "Severe earache or facial pain", "Difficulty breathing"]
    },
    "Pneumonia": {
        "description": "An acute inflammatory infection of the pulmonary parenchyma causing alveolar consolidation with exudate.",
        "urgency": "High",
        "specialist": "Pulmonologist",
        "precautions": [
            "Consult a doctor promptly for chest X-ray and pulse oximetry.",
            "Complete the full antibiotic or antiviral treatment as prescribed.",
            "Use an incentive spirometer or practice deep breathing exercises.",
            "Drink ample fluids and monitor blood oxygen saturation (SpO2)."
        ],
        "warning_signs": ["Oxygen saturation dropping below 92%", "Bluish tinge to lips or fingertips", "Severe pleuritic chest pain"]
    },
    "Hemorrhoids (Piles)": {
        "description": "Swollen and engorged vascular plexuses located in the lower rectum and anal canal.",
        "urgency": "Routine",
        "specialist": "Colorectal Surgeon / Proctologist",
        "precautions": [
            "Increase dietary fiber intake (whole grains, vegetables, psyllium husk).",
            "Drink at least 2 to 3 liters of water daily to soften stools.",
            "Take warm sitz baths for 15-20 minutes two to three times a day.",
            "Avoid straining during bowel movements or prolonged sitting on the toilet."
        ],
        "warning_signs": ["Significant bright red rectal bleeding causing weakness", "Severe anal pain indicating thrombosed hemorrhoid", "Fever"]
    },
    "Heart Attack": {
        "description": "Myocardial infarction: an acute coronary syndrome where coronary blood flow is blocked, causing myocardial ischemia and necrosis.",
        "urgency": "Emergency",
        "specialist": "Cardiologist / Emergency Medicine",
        "precautions": [
            "CALL EMERGENCY SERVICES (911/112) IMMEDIATELY - EVERY MINUTE COUNTS.",
            "Chew an uncoated adult aspirin (300 mg) if not allergic and advised by emergency responders.",
            "Have the patient sit down, remain calm, and loosen tight clothing.",
            "Prepare for CPR and AED use if the patient becomes unresponsive."
        ],
        "warning_signs": ["Crushing chest pain radiating to jaw, neck, left arm or back", "Shortness of breath with cold sweats", "Lightheadedness or syncope"]
    },
    "Varicose Veins": {
        "description": "Incompetent venous valves resulting in abnormally enlarged, dilated, and tortuous superficial veins, most common in the lower extremities.",
        "urgency": "Routine",
        "specialist": "Vascular Surgeon",
        "precautions": [
            "Wear graduated medical compression stockings as prescribed.",
            "Elevate your legs above the level of the heart when resting.",
            "Avoid prolonged standing or sitting without walking breaks.",
            "Engage in regular low-impact walking to stimulate calf muscle venous pump."
        ],
        "warning_signs": ["Sudden swelling, warmth, and redness in one calf (Deep Vein Thrombosis)", "Bleeding from a varicose vein", "Skin ulcers near ankles"]
    },
    "Hypothyroidism": {
        "description": "A clinical syndrome resulting from insufficient thyroid hormone production, slowing down the basal metabolic rate.",
        "urgency": "Routine",
        "specialist": "Endocrinologist",
        "precautions": [
            "Take prescribed levothyroxine consistently on an empty stomach 30-60 mins before breakfast.",
            "Schedule periodic serum TSH and Free T4 blood tests.",
            "Maintain a balanced, nutrient-dense diet rich in selenium and iodine (if appropriate).",
            "Do not stop or adjust medication dosages without physician review."
        ],
        "warning_signs": ["Severe hypothermia and drowsiness (Myxedema coma)", "Pericardial effusion", "Profound depression"]
    },
    "Hyperthyroidism": {
        "description": "An overactive thyroid gland producing excessive circulating thyroid hormones, accelerating systemic metabolism.",
        "urgency": "Moderate",
        "specialist": "Endocrinologist",
        "precautions": [
            "Take prescribed antithyroid medications (methimazole/PTU) or beta-blockers as directed.",
            "Avoid excessive dietary iodine (seaweed, kelp supplements).",
            "Manage physical stress and ensure adequate caloric intake.",
            "Report any sudden fever or sore throat immediately to your endocrinologist."
        ],
        "warning_signs": ["High fever, extreme agitation, tachycardia, delirium (Thyroid Storm)", "Protruding painful eyes (Graves orbitopathy)"]
    },
    "Hypoglycemia": {
        "description": "An acute drop in blood glucose concentration (<70 mg/dL), leading to sympathoadrenal activation and neuroglycopenia.",
        "urgency": "Moderate",
        "specialist": "Endocrinologist / General Physician",
        "precautions": [
            "Follow the 'Rule of 15': ingest 15 grams of fast-acting carbohydrate (e.g. 4 oz fruit juice or 3-4 glucose tablets).",
            "Re-check capillary blood glucose in 15 minutes; repeat if still under 70 mg/dL.",
            "Follow with a complex carbohydrate and protein snack once glucose normalizes.",
            "Always carry glucose tabs or candies if you take insulin or sulfonylureas."
        ],
        "warning_signs": ["Seizures", "Loss of consciousness", "Inability to swallow safely"]
    },
    "Osteoarthritis": {
        "description": "A degenerative joint disease involving progressive breakdown of articular cartilage and underlying subchondral bone remodeling.",
        "urgency": "Routine",
        "specialist": "Rheumatologist / Orthopedic Surgeon",
        "precautions": [
            "Engage in low-impact physical activities such as swimming, cycling, or water aerobics.",
            "Maintain an optimal body weight to minimize mechanical stress on weight-bearing joints.",
            "Apply heat packs to relieve morning stiffness and cold packs for acute swelling.",
            "Use assistive devices (braces, walking cane) if recommended by physical therapy."
        ],
        "warning_signs": ["Joint suddenly becomes red, hot, and intensely swollen (Septic arthritis)", "Inability to bear any weight"]
    },
    "Arthritis": {
        "description": "Inflammatory arthropathy characterized by chronic synovitis, pain, swelling, and reduced range of motion across synovial joints.",
        "urgency": "Routine",
        "specialist": "Rheumatologist",
        "precautions": [
            "Consult a rheumatologist for inflammatory markers (ESR, CRP, RF, anti-CCP).",
            "Balance gentle joint mobilization with scheduled rest periods.",
            "Consume an anti-inflammatory Mediterranean-style diet.",
            "Take prescribed disease-modifying antirheumatic drugs (DMARDs) reliably."
        ],
        "warning_signs": ["High fever with single swollen red joint", "Rapid systemic fatigue and weight loss"]
    },
    "Paroxysmal Positional Vertigo": {
        "description": "A peripheral vestibular disorder where displaced otoconia within the inner ear semicircular canals cause episodic spinning vertigo.",
        "urgency": "Routine",
        "specialist": "ENT Specialist / Neurologist",
        "precautions": [
            "Undergo the Epley or canalith repositioning maneuver performed by a specialist.",
            "Sit down immediately upon feeling dizzy to prevent falling or injury.",
            "Avoid abrupt head tilts, looking straight up, or bending rapidly forward.",
            "Sleep with your head slightly elevated on two pillows."
        ],
        "warning_signs": ["Vertigo accompanied by diplopia (double vision), facial numbness, or limb weakness", "Severe persistent vomiting"]
    },
    "Acne": {
        "description": "An inflammatory skin condition of pilosebaceous units triggered by sebum overproduction, follicular hyperkeratinization, and Cutibacterium acnes.",
        "urgency": "Routine",
        "specialist": "Dermatologist",
        "precautions": [
            "Wash affected areas twice daily with a gentle, non-comedogenic cleanser.",
            "Do not squeeze, pick, or pop pimples to prevent scarring and secondary infection.",
            "Use topical treatments (benzoyl peroxide, salicylic acid, retinoids) as advised.",
            "Apply non-greasy, oil-free moisturizer and broad-spectrum sunscreen."
        ],
        "warning_signs": ["Deep painful cystic nodules causing severe pain", "Systemic symptoms or fever"]
    },
    "Urinary Tract Infection (UTI)": {
        "description": "An infection affecting parts of the urinary tract (urethra, bladder, or kidneys), most commonly caused by uropathogenic Escherichia coli.",
        "urgency": "Routine",
        "specialist": "Urologist / General Physician",
        "precautions": [
            "Drink plenty of water to help flush bacteria out of the urinary tract.",
            "Complete the entire prescribed antibiotic regimen without skipping doses.",
            "Urinate promptly when the urge arises and after sexual intercourse.",
            "Avoid irritants like caffeine, alcohol, and spicy foods during active symptoms."
        ],
        "warning_signs": ["Flank or lower back pain with high fever and chills (Pyelonephritis)", "Visible blood in urine", "Vomiting"]
    },
    "Psoriasis": {
        "description": "A chronic autoimmune dermatological disorder causing hyperproliferation of keratinocytes, forming thick, erythematous silvery-scaled plaques.",
        "urgency": "Routine",
        "specialist": "Dermatologist",
        "precautions": [
            "Keep skin consistently moisturized with thick emollient ointments.",
            "Avoid skin trauma, scratching, or sunburn (which can trigger Koebner phenomenon).",
            "Manage mental stress through relaxation and mindfulness techniques.",
            "Discuss topical corticosteroids, vitamin D analogues, or biologic therapy with a dermatologist."
        ],
        "warning_signs": ["Erythroderma (rash covering >90% of body surface)", "Severe joint stiffness and swelling (Psoriatic arthritis)"]
    },
    "Impetigo": {
        "description": "A highly contagious superficial bacterial skin infection caused by Staphylococcus aureus or Streptococcus pyogenes, marked by honey-colored crusts.",
        "urgency": "Routine",
        "specialist": "Dermatologist / General Physician",
        "precautions": [
            "Gently cleanse sores with antibacterial wash and apply prescribed topical mupirocin.",
            "Wash hands thoroughly after touching affected areas.",
            "Keep sores loosely covered with sterile gauze to prevent spread.",
            "Wash clothing, bedding, and towels in hot water daily and do not share them."
        ],
        "warning_signs": ["Blisters spreading rapidly or accompanied by high fever", "Swollen tender lymph nodes", "Kidney pain / dark urine"]
    }
}


class MedicalKnowledgeBase:
    """Provides querying and graph retrieval for medical concepts and differential diagnosis."""

    def __init__(self, profiles: Optional[Dict] = None):
        self.knowledge = DISEASE_KNOWLEDGE
        self.profiles = profiles or {}

    def get_disease_info(self, disease_name: str) -> Dict:
        """Retrieve complete clinical metadata for a given disease."""
        info = self.knowledge.get(disease_name, {
            "description": f"Clinical profile for {disease_name}.",
            "urgency": "Moderate",
            "specialist": "General Physician",
            "precautions": [
                "Consult a licensed medical physician for formal clinical evaluation.",
                "Monitor your symptoms closely.",
                "Maintain adequate hydration and rest."
            ],
            "warning_signs": ["Difficulty breathing", "Loss of consciousness", "Severe unremitting pain"]
        })
        # Enrich with statistical hallmark symptoms if available
        if disease_name in self.profiles:
            info["hallmark_symptoms"] = self.profiles[disease_name].get("hallmark_symptoms", [])
        return info

    def get_urgency_level(self, disease_name: str) -> str:
        return self.get_disease_info(disease_name).get("urgency", "Moderate")

    def get_precautions(self, disease_name: str) -> List[str]:
        return self.get_disease_info(disease_name).get("precautions", [])

    def get_specialist(self, disease_name: str) -> str:
        return self.get_disease_info(disease_name).get("specialist", "General Physician")
