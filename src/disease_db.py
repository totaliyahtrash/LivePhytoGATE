"""PhytoGATE Local Disease Database.

Supplies deterministic agronomic treatment protocols and pathology profiles.
Completely decoupled from vision inference to ensure zero external API consumption
for treatment queries.
"""

from typing import Any, Dict, Optional
from src.taxonomy import validate_diagnosis

DISEASE_PROFILES: Dict[str, Dict[str, Any]] = {
    # Tomato Diseases
    "Tomato:Early Blight": {
        "host": "Tomato",
        "disease": "Early Blight",
        "pathogen_type": "Fungal (Alternaria solani / Alternaria linariae)",
        "severity_risk": "HIGH",
        "symptoms": [
            "Concentric target-like dark brown necrotic spots on older leaves",
            "Chlorotic yellow halo surrounding brown lesions",
            "Foliar yellowing and premature leaf senescence/defoliation",
            "Stem collar rot and sunken cankers on stems",
            "Dark, leathery sunken lesions at the stem end of fruits",
        ],
        "cultural_controls": [
            "Maintain 3- to 4-year crop rotation away from Solanaceae (potato, eggplant, pepper)",
            "Prune lowest foliage (bottom 12-18 inches) to eliminate soil-splash contact",
            "Apply drip irrigation at the base rather than overhead sprinkler irrigation",
            "Mulch heavily beneath plants with clean straw or plastic barrier",
            "Sanitize trellising stakes and tools between rows",
        ],
        "chemical_treatments": [
            "Chlorothalonil (Bravo / Daconil) applied at 7-10 day intervals as preventative protectant",
            "Azoxystrobin (Quadris) alternating with difenoconazole (Inspire Super) to reduce resistance",
            "Mancozeb or Copper hydroxide applied prior to rain events",
        ],
        "organic_treatments": [
            "Fixed copper fungicide (Copper Octanoate / Bordeaux mixture)",
            "Biofungicide sprays based on Bacillus amyloliquefaciens (Serenade ASO)",
            "Potassium bicarbonate (MilStop) for foliar pH suppression",
            "Cold-pressed neem oil applied at dawn or dusk",
        ],
        "prevention": [
            "Plant certified pathogen-free seeds and disease-resistant cultivars (e.g., 'Mountain Supreme', 'Defiant')",
            "Ensure wide spacing (24-36 inches) for rapid canopy drying and air circulation",
            "Immediately destroy or bury infected crop debris after harvest; do not compost",
        ],
    },
    "Tomato:Late Blight": {
        "host": "Tomato",
        "disease": "Late Blight",
        "pathogen_type": "Oomycete (Phytophthora infestans)",
        "severity_risk": "CRITICAL",
        "symptoms": [
            "Large, water-soaked, dark olive-brown to purple irregular foliar lesions",
            "Delicate white fungal-like sporulation on underside of leaves in humid conditions",
            "Rapid stem collapse and petiole girdling",
            "Firm, greasy brown marbling on green and ripe fruit",
        ],
        "cultural_controls": [
            "Strictly avoid overhead irrigation; keep foliage completely dry",
            "Promptly eradicate volunteer potato and tomato plants in early spring",
            "Bag and remove infected plants immediately on hot dry days to prevent spore dispersal",
        ],
        "chemical_treatments": [
            "Mandipropamid (Revus) or Cyazofamid (Ranman) targeted for Oomycetes",
            "Mefenoxam / Metalaxyl where strains remain sensitive",
            "Chlorothalonil or Mancozeb as standard preventative barriers",
        ],
        "organic_treatments": [
            "Bordeaux mixture or copper sulfate pentahydrate applied preventatively",
            "Bacillus subtilis foliar applications prior to disease onset",
        ],
        "prevention": [
            "Select resistant cultivars containing Ph-2 and Ph-3 resistance genes",
            "Monitor regional disease forecast networks for late blight outbreaks",
        ],
    },
    "Tomato:Bacterial Spot": {
        "host": "Tomato",
        "disease": "Bacterial Spot",
        "pathogen_type": "Bacterial (Xanthomonas perforans / Xanthomonas vesicatoria)",
        "severity_risk": "HIGH",
        "symptoms": [
            "Small (1-3mm), dark water-soaked circular to angular leaf spots",
            "Spots become brown-black with slightly raised borders and yellow halos",
            "Severe blighting and leaf drop exposing fruit to sunscald",
            "Small, raised scab-like rough black specks on fruit surface",
        ],
        "cultural_controls": [
            "Use hot-water treated or certified disease-free seeds",
            "Avoid handling, cultivating, or harvesting while foliage is wet",
            "Rotate crops for at least 2-3 years with non-hosts (corn, beans)",
        ],
        "chemical_treatments": [
            "Fixed copper combined with Mancozeb (improves copper solubility against resistant strains)",
            "Acibenzolar-S-methyl (Actigard) as systemic acquired resistance inducer",
        ],
        "organic_treatments": [
            "Copper bactericides formulated with basic copper sulfate",
            "Bacteriophage biopesticides (AgriPhage)",
            "Streptomyces lydicus or Bacillus subtilis biofungicides",
        ],
        "prevention": [
            "Eliminate solanaceous weeds (nightshade, horsenettle) around field margins",
            "Maintain clean drip systems and sanitize greenhouse transplant trays",
        ],
    },
    "Tomato:Septoria Leaf Spot": {
        "host": "Tomato",
        "disease": "Septoria Leaf Spot",
        "pathogen_type": "Fungal (Septoria lycopersici)",
        "severity_risk": "MODERATE",
        "symptoms": [
            "Numerous small, circular lesions (1.5-3mm) with ash-gray centers and dark brown borders",
            "Tiny black pycnidia (fruiting bodies) visible with 10x lens inside gray centers",
            "Lesions start on oldest bottom leaves and progressively work up the canopy",
            "Heavily infected leaves turn yellow, wither, and drop off",
        ],
        "cultural_controls": [
            "Strip bottom leaves once plants reach 24 inches to limit splash inoculation",
            "Provide thick organic mulch (wood chips, straw) under canopies",
            "Maintain wide vine spacing and trellis vertically for maximum aeration",
        ],
        "chemical_treatments": [
            "Chlorothalonil (Daconil) applied on 7-day schedule from early emergence",
            "Pyraclostrobin (Cabrio) or Boscalid (Endura)",
        ],
        "organic_treatments": [
            "Liquid copper octanoate every 7 to 10 days",
            "Bio-protectants containing Bacillus pumilus or Trichoderma harzianum",
        ],
        "prevention": [
            "Complete fall tillage to bury crop residue and accelerate decomposition",
            "Two-year minimum rotation without tomatoes, potatoes, or eggplants",
        ],
    },
    "Tomato:Leaf Mold": {
        "host": "Tomato",
        "disease": "Leaf Mold",
        "pathogen_type": "Fungal (Passalora fulva / Cladosporium fulvum)",
        "severity_risk": "MODERATE",
        "symptoms": [
            "Pale green to yellowish indistinct patches on upper leaf surface",
            "Olive-green to velvety brown mold growth on corresponding lower leaf surface",
            "Infected leaves curl, wither, and drop prematurely",
        ],
        "cultural_controls": [
            "Increase ventilation and airflow in high tunnels/greenhouses to keep relative humidity below 85%",
            "Space plants generously to promote rapid air circulation",
        ],
        "chemical_treatments": [
            "Difenoconazole, Cyazofamid, or Boscalid",
            "Chlorothalonil applied as preventative canopy spray",
        ],
        "organic_treatments": [
            "Copper hydroxide sprays",
            "Potassium bicarbonate foliar wash",
        ],
        "prevention": [
            "Select resistant varieties with Cf genes",
            "Heat high tunnels and vent moist evening air",
        ],
    },
    "Tomato:Target Spot": {
        "host": "Tomato",
        "disease": "Target Spot",
        "pathogen_type": "Fungal (Corynespora cassiicola)",
        "severity_risk": "HIGH",
        "symptoms": [
            "Pinpoint brown lesions enlarging into circular spots with light brown centers",
            "Distinct concentric rings resembling early blight but without broad yellow halos",
            "Deep sunken crater lesions on fruit with dark centers",
        ],
        "cultural_controls": [
            "Eliminate weed hosts such as morningglory and soybean residue",
            "Avoid overhead irrigation and excessive canopy density",
        ],
        "chemical_treatments": [
            "Fluopyram + Trifloxystrobin (Luna Sensation)",
            "Famoxadone + Cymoxanil (Tanos)",
        ],
        "organic_treatments": [
            "Copper formulations combined with biological bio-fungicides",
        ],
        "prevention": [
            "Standard multi-year crop rotation and strict post-harvest sanitation",
        ],
    },
    "Tomato:Spider Mites": {
        "host": "Tomato",
        "disease": "Spider Mites",
        "pathogen_type": "Acarid Pest (Tetranychus urticae)",
        "severity_risk": "MODERATE",
        "symptoms": [
            "Fine pale yellow stippling or bronzing on upper leaf surface",
            "Silken webbing on leaf undersides and terminal growing tips",
            "Brittle, desiccated foliage leading to rapid defoliation under dry, dusty conditions",
        ],
        "cultural_controls": [
            "Suppress dust along field roadways (overhead dust favors mite outbreaks)",
            "Wash plants with water jets to disrupt webs and dislodge colonies",
        ],
        "chemical_treatments": [
            "Bifenazate (Acramite) or Spiromesifen (Oberon)",
            "Abamectin (Agri-Mek) targeted to leaf undersides",
        ],
        "organic_treatments": [
            "Insecticidal soap (potassium salts of fatty acids)",
            "Horticultural mineral oils or pure cold-pressed neem oil",
            "Release predatory mites (Phytoseiulus persimilis)",
        ],
        "prevention": [
            "Avoid broad-spectrum synthetic pyrethroids that eliminate natural mite predators",
            "Consistently monitor field edges during hot, drought periods",
        ],
    },
    "Tomato:Tomato Yellow Leaf Curl Virus": {
        "host": "Tomato",
        "disease": "Tomato Yellow Leaf Curl Virus",
        "pathogen_type": "Viral (Begomovirus / TYLCV - Whitefly transmitted)",
        "severity_risk": "HIGH",
        "symptoms": [
            "Marked stunting of plants and bushy upright growth habit",
            "Upward curling, cupping, and crinkling of leaf margins",
            "Interveinal chlorosis with reduced leaf blade size",
            "Severe flower abscission and drastic fruit drop",
        ],
        "cultural_controls": [
            "Deploy yellow sticky cards for whitefly population tracking",
            "Install insect-exclusion screens (50-mesh) in nurseries and high tunnels",
            "Rogue out and destroy infected symptomatic plants immediately",
        ],
        "chemical_treatments": [
            "Target vector (Bemisia tabaci) using Imidacloprid or Thiamethoxam",
            "Flupyradifurone (Sivanto) or Spirotetramat (Movento)",
        ],
        "organic_treatments": [
            "Beauveria bassiana entomopathogenic fungal bio-insecticide",
            "Pyrethrin mixed with insecticidal soap for knockdown",
        ],
        "prevention": [
            "Plant TYLCV-resistant hybrids carrying Ty-1 or Ty-3 resistance genes",
            "Maintain a regional whitefly-free host-free period between planting seasons",
        ],
    },
    "Tomato:Tomato Mosaic Virus": {
        "host": "Tomato",
        "disease": "Tomato Mosaic Virus",
        "pathogen_type": "Viral (Tobamovirus / ToMV - Mechanically transmitted)",
        "severity_risk": "HIGH",
        "symptoms": [
            "Mottling with alternating light and dark green mosaic patterns on foliage",
            "Fern-like leaf distortion and blistering",
            "Internal brown necrosis (browning) of fruit walls",
        ],
        "cultural_controls": [
            "Decontaminate hands and pruning knives with 20% non-fat dry milk or 10% trisodium phosphate",
            "Prohibit tobacco use near greenhouse or field crops (virus is tobacco-transmissible)",
        ],
        "chemical_treatments": [
            "No chemical viricides exist; vector and tool sanitation is mandatory",
        ],
        "organic_treatments": [
            "Dip tools in skim milk solution during pruning to deactivate viral particles",
        ],
        "prevention": [
            "Plant resistant varieties possessing the Tm-2 or Tm-2^2 resistance alleles",
            "Strictly purchase certified virus-indexed seeds",
        ],
    },
    "Tomato:Healthy": {
        "host": "Tomato",
        "disease": "Healthy",
        "pathogen_type": "None (Healthy Foliage)",
        "severity_risk": "NONE",
        "symptoms": [
            "Vibrant, uniform green coloration without necrotic spots or chlorosis",
            "Turgid leaves with intact margins and natural serration",
            "Normal vegetative vigor and balanced internodal spacing",
        ],
        "cultural_controls": [
            "Continue standard drip irrigation and balanced fertilization (N-P-K 5-10-10)",
            "Maintain mulch barrier and stake support",
        ],
        "chemical_treatments": [
            "No chemical intervention required",
        ],
        "organic_treatments": [
            "Routine foliar seaweed or compost tea extracts for vitality",
        ],
        "prevention": [
            "Regular weekly scouting of lower canopy to detect early pathogen arrival",
        ],
    },

    # Potato Diseases
    "Potato:Early Blight": {
        "host": "Potato",
        "disease": "Early Blight",
        "pathogen_type": "Fungal (Alternaria solani)",
        "severity_risk": "HIGH",
        "symptoms": [
            "Small dark brown to black spots with concentric rings (target pattern) on lower leaves",
            "Yellow halo around lesions leading to premature leaf drop",
            "Brown, circular dry rot lesions on potato tubers",
        ],
        "cultural_controls": [
            "Ensure balanced nitrogen fertility (plants under nitrogen stress are more vulnerable)",
            "Hill potatoes well to shield developing tubers from fungal spores washed off leaves",
        ],
        "chemical_treatments": [
            "Chlorothalonil, Mancozeb, or Azoxystrobin applied per blight forecast index",
        ],
        "organic_treatments": [
            "Copper sulfate or copper octanoate applied preventatively",
            "Bacillus amyloliquefaciens foliar treatments",
        ],
        "prevention": [
            "Use certified seed tubers; rotate with non-solanaceous crops for 3 years",
        ],
    },
    "Potato:Late Blight": {
        "host": "Potato",
        "disease": "Late Blight",
        "pathogen_type": "Oomycete (Phytophthora infestans)",
        "severity_risk": "CRITICAL",
        "symptoms": [
            "Large, dark water-soaked lesions that rapidly expand across foliage and stems",
            "White velvety sporulation on leaf underside in moist cool conditions",
            "Tubers develop dry, granular reddish-brown flesh rot",
        ],
        "cultural_controls": [
            "Destroy all cull piles and volunteer plants before planting season",
            "Kill vines 2-3 weeks before harvest to prevent tuber contamination during digging",
        ],
        "chemical_treatments": [
            "Mandipropamid, Fluopicolide, or Cyazofamid",
            "Protectant sprays with Mancozeb or Chlorothalonil",
        ],
        "organic_treatments": [
            "Copper hydroxide formulations applied at 5-7 day intervals",
        ],
        "prevention": [
            "Plant resistant seed varieties and follow regional late blight warning alerts",
        ],
    },
    "Potato:Healthy": {
        "host": "Potato",
        "disease": "Healthy",
        "pathogen_type": "None (Healthy Foliage)",
        "severity_risk": "NONE",
        "symptoms": [
            "Uniform green leaf canopy without necrotic spots, halos, or wilt",
        ],
        "cultural_controls": [
            "Maintain consistent hilling and regular moisture levels",
        ],
        "chemical_treatments": ["None required"],
        "organic_treatments": ["None required"],
        "prevention": ["Scout fields weekly, especially following rain events"],
    },

    # Apple Diseases
    "Apple:Apple Scab": {
        "host": "Apple",
        "disease": "Apple Scab",
        "pathogen_type": "Fungal (Venturia inaequalis)",
        "severity_risk": "HIGH",
        "symptoms": [
            "Olive-green to velvety brown-black spots on leaves and fruit",
            "Leaves turn yellow and drop prematurely",
            "Fruit develops corky, scabby lesions that crack and distort",
        ],
        "cultural_controls": [
            "Shred and rake fallen orchard leaves in autumn to accelerate breakdown of overwintering ascospores",
            "Prune canopy for sunlight penetration and rapid air circulation",
        ],
        "chemical_treatments": [
            "Captan or Mancozeb as primary protectants during green tip through petal fall",
            "Myclobutanil or Difenoconazole for post-infection reach-back activity",
        ],
        "organic_treatments": [
            "Liquid lime sulfur or sulfur wettable powder",
            "Potassium bicarbonate sprays",
        ],
        "prevention": [
            "Plant scab-resistant cultivars ('Liberty', 'Enterprise', 'Freedom', 'GoldRush')",
        ],
    },
    "Apple:Black Rot": {
        "host": "Apple",
        "disease": "Black Rot",
        "pathogen_type": "Fungal (Botryosphaeria obtusa)",
        "severity_risk": "MODERATE",
        "symptoms": [
            "'Frog-eye' leaf spots: circular lesions with purple margins and tan centers",
            "Firm brown rot on fruit that turns completely black and mummifies",
            "Bark cankers on branches and limbs",
        ],
        "cultural_controls": [
            "Prune out dead wood, mummified fruits, and fire blight strikes during winter",
        ],
        "chemical_treatments": [
            "Captan, Mancozeb, or Thiophanate-methyl sprays",
        ],
        "organic_treatments": [
            "Copper formulations applied at dormant or delayed-dormant stages",
        ],
        "prevention": [
            "Remove all mummies from orchard canopy before spring bud break",
        ],
    },
    "Apple:Cedar Apple Rust": {
        "host": "Apple",
        "disease": "Cedar Apple Rust",
        "pathogen_type": "Fungal (Gymnosporangium juniperi-virginianae)",
        "severity_risk": "MODERATE",
        "symptoms": [
            "Bright orange-yellow circular spots on upper surface of leaves",
            "Small black dots inside orange spots, followed by tube-like aecia on underside",
        ],
        "cultural_controls": [
            "Remove nearby Eastern red cedar (Juniperus virginiana) trees within a 1-2 mile radius if feasible",
        ],
        "chemical_treatments": [
            "Myclobutanil (Immunox) applied from pink bud stage through early cover",
        ],
        "organic_treatments": [
            "Sulfur sprays applied before rain events when cedar galls are gelatinizing",
        ],
        "prevention": [
            "Select resistant cultivars ('Liberty', 'Freedom', 'Enterprise')",
        ],
    },
    "Apple:Healthy": {
        "host": "Apple",
        "disease": "Healthy",
        "pathogen_type": "None (Healthy Foliage)",
        "severity_risk": "NONE",
        "symptoms": ["Uniform green leaf surface without lesion, scab, or rust blemishes"],
        "cultural_controls": ["Maintain dormant season pruning and balanced nutrition"],
        "chemical_treatments": ["None required"],
        "organic_treatments": ["None required"],
        "prevention": ["Seasonal orchard monitoring"],
    },

    # Corn Diseases
    "Corn:Cercospora Leaf Spot (Gray Leaf Spot)": {
        "host": "Corn",
        "disease": "Cercospora Leaf Spot (Gray Leaf Spot)",
        "pathogen_type": "Fungal (Cercospora zeae-maydis)",
        "severity_risk": "HIGH",
        "symptoms": [
            "Small tan spots expanding into long, narrow rectangular lesions delimited by leaf veins",
            "Lesions turn grayish as spores develop, leading to extensive blighting of canopy",
        ],
        "cultural_controls": [
            "Rotate out of corn for 1-2 years to allow residue decomposition",
            "Utilize conservation tillage to bury crop debris where erosion allows",
        ],
        "chemical_treatments": [
            "Pyraclostrobin + Fluxapyroxad (Priaxor) or Azoxystrobin + Propiconazole (Quilt Xcel)",
        ],
        "organic_treatments": [
            "Copper fungicides where compatible with hybrid programs",
        ],
        "prevention": [
            "Select hybrids with high GLS resistance ratings",
        ],
    },
    "Corn:Common Rust": {
        "host": "Corn",
        "disease": "Common Rust",
        "pathogen_type": "Fungal (Puccinia sorghi)",
        "severity_risk": "MODERATE",
        "symptoms": [
            "Golden-brown to cinnamon-brown powdery pustules scattered over both leaf surfaces",
            "Pustules turn brownish-black late in the season",
        ],
        "cultural_controls": [
            "Plant early to avoid late-season airborne spore migrations",
        ],
        "chemical_treatments": [
            "Triazole or strobilurin fungicides if pustules appear before silking",
        ],
        "organic_treatments": [
            "Foliar biofungicides (Bacillus-based) applied early",
        ],
        "prevention": [
            "Use hybrids containing Rp resistance genes",
        ],
    },
    "Corn:Northern Leaf Blight": {
        "host": "Corn",
        "disease": "Northern Leaf Blight",
        "pathogen_type": "Fungal (Exserohilum turcicum)",
        "severity_risk": "HIGH",
        "symptoms": [
            "Long, elliptical, cigar-shaped grayish-green to tan lesions (1-6 inches long)",
            "Lesions coalesce under humid conditions, giving plants a frost-killed appearance",
        ],
        "cultural_controls": [
            "Two-year rotation with soybeans or small grains",
            "Residue management through fall tillage",
        ],
        "chemical_treatments": [
            "Azoxystrobin + Difenoconazole or Pyraclostrobin at tassel (VT) stage",
        ],
        "organic_treatments": [
            "Foliar sulfur or copper protectants",
        ],
        "prevention": [
            "Plant resistant hybrids carrying Ht1, Ht2, or Ht3 resistance genes",
        ],
    },
    "Corn:Healthy": {
        "host": "Corn",
        "disease": "Healthy",
        "pathogen_type": "None (Healthy Foliage)",
        "severity_risk": "NONE",
        "symptoms": ["Smooth, emerald-green broad leaves without streaks or pustules"],
        "cultural_controls": ["Maintain adequate nitrogen side-dressing"],
        "chemical_treatments": ["None required"],
        "organic_treatments": ["None required"],
        "prevention": ["Routine field walking"],
    },

    # Grape Diseases
    "Grape:Black Rot": {
        "host": "Grape",
        "disease": "Black Rot",
        "pathogen_type": "Fungal (Guignardia bidwellii)",
        "severity_risk": "HIGH",
        "symptoms": [
            "Small circular reddish-brown leaf spots with tiny black dots around border",
            "Berries shrivel into hard, black, wrinkled mummies that cling to cluster",
        ],
        "cultural_controls": [
            "Remove and bury all mummified berries during winter pruning",
            "Canopy management: shoot thinning and leaf pulling to promote ventilation",
        ],
        "chemical_treatments": [
            "Mancozeb or Ziram applied from bud break through fruit set",
            "Myclobutanil or Tebuconazole for systemic control",
        ],
        "organic_treatments": [
            "Copper hydroxide combined with sulfur",
        ],
        "prevention": [
            "Maintain strict vineyard sanitation and open trellising",
        ],
    },
    "Grape:Esca (Black Measles)": {
        "host": "Grape",
        "disease": "Esca (Black Measles)",
        "pathogen_type": "Fungal Complex (Phaeomoniella chlamydospora, Fomitiporia mediterranea)",
        "severity_risk": "HIGH",
        "symptoms": [
            "'Tiger-stripe' chlorotic and necrotic patterns between leaf veins",
            "Dark purple/brown spots on berries resembling measles",
            "Sudden vine collapse (apoplexy) during hot spells",
        ],
        "cultural_controls": [
            "Avoid pruning wounds during wet weather; apply wound sealants to cuts",
            "Retrain suckers to replace severely damaged cordons or trunks",
        ],
        "chemical_treatments": [
            "Pruning wound protectants containing tebuconazole or thiophanate-methyl",
        ],
        "organic_treatments": [
            "Trichoderma-based biocontrol pruning wound dressings",
        ],
        "prevention": [
            "Double-pruning techniques in late winter",
        ],
    },
    "Grape:Leaf Blight (Isariopsis Leaf Spot)": {
        "host": "Grape",
        "disease": "Leaf Blight (Isariopsis Leaf Spot)",
        "pathogen_type": "Fungal (Pseudocercospora vitis)",
        "severity_risk": "MODERATE",
        "symptoms": [
            "Angular to irregular dark brown lesions with dark purplish-brown borders",
            "Premature defoliation late in summer reducing sugar accumulation in grapes",
        ],
        "cultural_controls": [
            "Ensure proper vine ventilation and shoot positioning",
        ],
        "chemical_treatments": [
            "Copper formulations, Mancozeb, or Captan",
        ],
        "organic_treatments": [
            "Bordeaux mixture applied at post-bloom stages",
        ],
        "prevention": [
            "Vineyard air drainage maintenance",
        ],
    },
    "Grape:Healthy": {
        "host": "Grape",
        "disease": "Healthy",
        "pathogen_type": "None (Healthy Foliage)",
        "severity_risk": "NONE",
        "symptoms": ["Lush palmate green leaves without chlorotic banding or necrotic lesions"],
        "cultural_controls": ["Standard canopy training and deficit irrigation"],
        "chemical_treatments": ["None required"],
        "organic_treatments": ["None required"],
        "prevention": ["Regular canopy scouting"],
    },

    # Pepper Diseases
    "Pepper:Bacterial Spot": {
        "host": "Pepper",
        "disease": "Bacterial Spot",
        "pathogen_type": "Bacterial (Xanthomonas euvesicatoria)",
        "severity_risk": "HIGH",
        "symptoms": [
            "Small water-soaked circular to irregular dark spots on leaves",
            "Severe leaf drop causing exposed fruit to sunscald",
            "Rough, raised warty circular spots on pepper fruit",
        ],
        "cultural_controls": [
            "Plant certified pathogen-free seed",
            "Drip irrigation only; avoid field operations when leaves are wet",
        ],
        "chemical_treatments": [
            "Copper hydroxide mixed with Mancozeb",
            "Actigard (systemic acquired resistance inducer)",
        ],
        "organic_treatments": [
            "Fixed copper sprays and bio-bactericides (Bacillus subtilis)",
        ],
        "prevention": [
            "Plant varieties with Bs2 / Bs3 resistance genes",
        ],
    },
    "Pepper:Frogeye Leaf Spot": {
        "host": "Pepper",
        "disease": "Frogeye Leaf Spot",
        "pathogen_type": "Fungal (Cercospora capsici)",
        "severity_risk": "HIGH",
        "symptoms": [
            "Small circular to oblong foliar lesions with light gray or tan centers and dark brown-to-purple margins (frog-eye appearance)",
            "Older lesions may crack or centers drop out, leaving a shot-hole appearance",
            "Extensive chlorosis surrounding lesions causing premature defoliation and flower/fruit drop",
            "Stems and petioles may develop elongated dark lesions during severe disease pressure",
        ],
        "cultural_controls": [
            "Rotate out of Solanaceous crops (pepper, tomato, eggplant) for at least 2 years",
            "Use drip irrigation to prevent water splashing and maintain dry foliage",
            "Remove and destroy infected plant debris immediately after harvest; do not compost",
            "Increase plant spacing to improve canopy ventilation and reduce relative humidity",
        ],
        "chemical_treatments": [
            "Chlorothalonil (Bravo / Daconil) applied preventatively at 7-14 day intervals",
            "Azoxystrobin (Quadris) alternating with Difenoconazole (Inspire Super) to mitigate resistance",
            "Copper hydroxide plus Mancozeb tank mixes for broad-spectrum protectant barrier",
        ],
        "organic_treatments": [
            "Fixed copper sprays (copper octanoate / basic copper sulfate)",
            "Biofungicide sprays formulated with Bacillus amyloliquefaciens or Bacillus subtilis",
            "Cold-pressed neem oil applied at low light (early morning or dusk)",
        ],
        "prevention": [
            "Plant certified disease-free, high-germination seed or pathogen-free transplants",
            "Avoid overhead irrigation, particularly late in the afternoon or evening",
            "Monitor canopy weekly during warm (75-85°F), humid or rainy conditions",
        ],
    },
    "Pepper:Healthy": {
        "host": "Pepper",
        "disease": "Healthy",
        "pathogen_type": "None (Healthy Foliage)",
        "severity_risk": "NONE",
        "symptoms": ["Glossy deep-green leaves with smooth margins and no blemishes"],
        "cultural_controls": ["Balanced fertility and regular soil moisture"],
        "chemical_treatments": ["None required"],
        "organic_treatments": ["None required"],
        "prevention": ["Weekly foliar inspection"],
    },

    # Rice Diseases
    "Rice:Bacterial Leaf Blight": {
        "host": "Rice",
        "disease": "Bacterial Leaf Blight",
        "pathogen_type": "Bacterial (Xanthomonas oryzae pv. oryzae)",
        "severity_risk": "CRITICAL",
        "symptoms": [
            "Water-soaked stripes starting at leaf tips and margins",
            "Lesions turn yellow to grayish-white with wavy margins",
            "Bacterial ooze droplets visible on young lesions in morning dew",
        ],
        "cultural_controls": [
            "Avoid high nitrogen fertilization which increases susceptibility",
            "Maintain controlled water depth and avoid flooding from diseased paddies",
        ],
        "chemical_treatments": [
            "Copper oxychloride or bactericide applications where registered",
        ],
        "organic_treatments": [
            "Bio-formulations of Pseudomonas fluorescens",
        ],
        "prevention": [
            "Plant resistant rice varieties carrying Xa genes (e.g., Xa21)",
        ],
    },
    "Rice:Brown Spot": {
        "host": "Rice",
        "disease": "Brown Spot",
        "pathogen_type": "Fungal (Bipolaris oryzae)",
        "severity_risk": "HIGH",
        "symptoms": [
            "Small circular to oval brown spots with gray or whitish centers and yellow halo",
            "Spots cover leaf blade, sheath, and panicle branches, causing grain discoloration",
        ],
        "cultural_controls": [
            "Correct soil nutrient deficiencies (potassium, silicon, and zinc)",
        ],
        "chemical_treatments": [
            "Propiconazole, Mancozeb, or Tricyclazole sprays",
        ],
        "organic_treatments": [
            "Seed treatment with bio-control agents (Trichoderma viride)",
        ],
        "prevention": [
            "Use certified clean seed and balance soil fertility",
        ],
    },
    "Rice:Rice Blast": {
        "host": "Rice",
        "disease": "Rice Blast",
        "pathogen_type": "Fungal (Magnaporthe oryzae)",
        "severity_risk": "CRITICAL",
        "symptoms": [
            "Diamond-shaped or spindle-shaped lesions with gray centers and dark reddish-brown borders",
            "Neck rot / collar rot causing panicles to fall over and produce empty white heads",
        ],
        "cultural_controls": [
            "Avoid excessive nitrogen; maintain adequate field flooding",
        ],
        "chemical_treatments": [
            "Tricyclazole, Isoprothiolane, or Azoxystrobin",
        ],
        "organic_treatments": [
            "Silicon soil amendments and botanical extracts",
        ],
        "prevention": [
            "Plant blast-resistant certified cultivars",
        ],
    },
    "Rice:Healthy": {
        "host": "Rice",
        "disease": "Healthy",
        "pathogen_type": "None (Healthy Foliage)",
        "severity_risk": "NONE",
        "symptoms": ["Uniform green upright blades without lesions, blast diamonds, or spots"],
        "cultural_controls": ["Balanced water management and fertilization"],
        "chemical_treatments": ["None required"],
        "organic_treatments": ["None required"],
        "prevention": ["Paddy scouting"],
    },

    # Wheat Diseases
    "Wheat:Leaf Rust": {
        "host": "Wheat",
        "disease": "Leaf Rust",
        "pathogen_type": "Fungal (Puccinia triticina)",
        "severity_risk": "HIGH",
        "symptoms": [
            "Small, round-to-oval orange-brown pustules scattered across upper leaf surfaces",
            "Pustules rupture epidermal tissue, releasing powdery urediniospores",
        ],
        "cultural_controls": [
            "Eradicate volunteer wheat plants before autumn sowing",
        ],
        "chemical_treatments": [
            "Triazole fungicides (Propiconazole, Tebuconazole) at flag leaf emergence",
        ],
        "organic_treatments": [
            "Sulfur or biocontrol sprays where suitable",
        ],
        "prevention": [
            "Plant resistant cultivars carrying Lr resistance genes",
        ],
    },
    "Wheat:Powdery Mildew": {
        "host": "Wheat",
        "disease": "Powdery Mildew",
        "pathogen_type": "Fungal (Blumeria graminis f. sp. tritici)",
        "severity_risk": "MODERATE",
        "symptoms": [
            "White to light gray powdery patches on leaf blades and sheaths",
            "Patches turn dull gray-brown with tiny black chasmothecia embedded",
        ],
        "cultural_controls": [
            "Avoid overly dense seeding rates that trap humidity",
        ],
        "chemical_treatments": [
            "Metconazole, Pyraclostrobin, or Prothioconazole",
        ],
        "organic_treatments": [
            "Potassium bicarbonate or sulfur foliar sprays",
        ],
        "prevention": [
            "Use varieties with Pm resistance genes",
        ],
    },
    "Wheat:Stripe Rust": {
        "host": "Wheat",
        "disease": "Stripe Rust",
        "pathogen_type": "Fungal (Puccinia striiformis)",
        "severity_risk": "CRITICAL",
        "symptoms": [
            "Yellow-orange pustules arranged in prominent linear stripes along leaf veins",
            "Attacks leaves and glumes, causing rapid shriveling in cool moist weather",
        ],
        "cultural_controls": [
            "Timely planting and destruction of green bridges",
        ],
        "chemical_treatments": [
            "Azoxystrobin + Propiconazole applied early upon detection",
        ],
        "organic_treatments": [
            "Early sulfur dust applications",
        ],
        "prevention": [
            "Plant stripe-rust resistant cultivars with Yr genes",
        ],
    },
    "Wheat:Healthy": {
        "host": "Wheat",
        "disease": "Healthy",
        "pathogen_type": "None (Healthy Foliage)",
        "severity_risk": "NONE",
        "symptoms": ["Vigorous green leaf blades without pustules, stripes, or powdery growth"],
        "cultural_controls": ["Balanced N-P-K management"],
        "chemical_treatments": ["None required"],
        "organic_treatments": ["None required"],
        "prevention": ["Flag-leaf scouting"],
    },

    # Soybean Diseases
    "Soybean:Frogeye Leaf Spot": {
        "host": "Soybean",
        "disease": "Frogeye Leaf Spot",
        "pathogen_type": "Fungal (Cercospora sojina)",
        "severity_risk": "HIGH",
        "symptoms": [
            "Small, circular to angular lesions on upper leaf surfaces",
            "Lesions have dark reddish-brown margins with light brown to ash-gray centers",
            "Clusters of dark conidiophores visible in lesion centers under high humidity",
            "Lesions coalesce causing blighting and premature defoliation in severe outbreaks",
        ],
        "cultural_controls": [
            "Implement a 1- to 2-year crop rotation with non-hosts (corn, sorghum, small grains)",
            "Perform residue management / tillage to bury infected soybean stubble where conservation plans permit",
            "Plant certified pathogen-free, fungicide-treated seed",
        ],
        "chemical_treatments": [
            "Apply multi-mode of action premixes at R3 (beginning pod) growth stage",
            "Triazole fungicides (Difenoconazole, Prothioconazole, Tetraconazole)",
            "SDHI fungicides (Fluxapyroxad, Benzovindiflupyr) combined with Strobilurins to manage QoI resistance",
        ],
        "organic_treatments": [
            "Preventative copper hydroxide or copper octanoate foliar sprays",
            "Biofungicide sprays based on Bacillus amyloliquefaciens or Bacillus subtilis",
            "Botanical extract sprays (Reynoutria sachalinensis)",
        ],
        "prevention": [
            "Select resistant soybean cultivars carrying the Rcs3 resistance gene",
            "Scout canopy during reproductive stages R1 through R5, particularly in warm humid weather",
        ],
    },
    "Soybean:Bacterial Blight": {
        "host": "Soybean",
        "disease": "Bacterial Blight",
        "pathogen_type": "Bacterial (Pseudomonas savastanoi pv. glycinea)",
        "severity_risk": "MODERATE",
        "symptoms": [
            "Small, angular, water-soaked yellow to brown spots with yellow halos",
            "Lesion centers dry out and drop out, giving leaves a tattered or shredded appearance",
            "Symptoms concentrated on young leaves in the upper canopy following cool, wet, windy storms",
        ],
        "cultural_controls": [
            "Avoid cultivating or operating machinery in fields when soybean foliage is wet",
            "Rotate crops for at least 1 year away from soybeans",
        ],
        "chemical_treatments": [
            "Copper bactericides applied preventatively before storms if disease history exists",
        ],
        "organic_treatments": [
            "Fixed copper sprays and bio-bactericide formulations",
        ],
        "prevention": [
            "Plant certified clean seed and resistant cultivars; avoid overhead irrigation",
        ],
    },
    "Soybean:Septoria Brown Spot": {
        "host": "Soybean",
        "disease": "Septoria Brown Spot",
        "pathogen_type": "Fungal (Septoria glycines)",
        "severity_risk": "MODERATE",
        "symptoms": [
            "Irregular, dark brown spots on lower leaves that enlarge and turn surrounding tissue yellow",
            "Premature defoliation starting at the base of the plant and moving upward",
        ],
        "cultural_controls": [
            "Crop rotation with corn or small grains for 1-2 years",
            "Deep tillage to accelerate decomposition of infected plant debris",
        ],
        "chemical_treatments": [
            "Fungicide application at R3 (early pod set) if weather remains wet and disease climbs into mid-canopy",
        ],
        "organic_treatments": [
            "Preventative copper applications",
        ],
        "prevention": [
            "Ensure good field drainage and avoid continuous soybean monoculture",
        ],
    },
    "Soybean:Healthy": {
        "host": "Soybean",
        "disease": "Healthy",
        "pathogen_type": "None (Healthy Foliage)",
        "severity_risk": "NONE",
        "symptoms": ["Trifoliate leaves with uniform rich green color, intact margins, and no lesions"],
        "cultural_controls": ["Standard soil fertility, weed management, and nodulation monitoring"],
        "chemical_treatments": ["None required"],
        "organic_treatments": ["None required"],
        "prevention": ["Routine weekly scouting across vegetative and reproductive stages"],
    },
}


def get_disease_profile(host: Optional[str], disease: Optional[str]) -> Optional[Dict[str, Any]]:
    """Retrieves the agronomic treatment and management profile for a validated host and disease.

    This function is completely local and deterministically retrieves information
    from DISEASE_PROFILES. Zero external AI calls are made.
    """
    if not host or not disease:
        return None

    # Validate against taxonomy first
    is_valid, canon_host, canon_disease = validate_diagnosis(host, disease)
    if not is_valid or not canon_host or not canon_disease:
        return None

    # Try exact key "Host:Disease"
    key = f"{canon_host}:{canon_disease}"
    if key in DISEASE_PROFILES:
        return DISEASE_PROFILES[key]

    # Return standard structured fallback profile for supported taxonomy missing profile
    return {
        "host": canon_host,
        "disease": canon_disease,
        "pathogen_type": "Agronomic Plant Pathogen",
        "severity_risk": "MODERATE",
        "symptoms": [
            f"Foliar lesions and tissue discoloration characteristic of {canon_disease} on {canon_host}."
        ],
        "cultural_controls": [
            "Maintain crop rotation away from host families.",
            "Avoid overhead irrigation to keep foliage dry.",
            "Remove and properly dispose of infected foliage.",
        ],
        "chemical_treatments": [
            "Apply registered broad-spectrum fungicides/bactericides according to local extension guidelines."
        ],
        "organic_treatments": [
            "Fixed copper fungicide or biofungicide (Bacillus subtilis) sprays.",
            "Neem oil foliar treatment."
        ],
        "prevention": [
            "Select resistant cultivars and ensure adequate vine spacing for air circulation."
        ],
    }
