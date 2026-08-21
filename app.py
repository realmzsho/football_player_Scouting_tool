import streamlit as st
import pandas as pd
import numpy as np
import joblib


"""
Core logic for the Player Scout app.
Extracted & adapted from the original Colab notebook.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

name_col = 'short_name'


def load_and_prepare(csv_path):
    """Loads the raw csv and runs the full preprocessing / feature engineering
    / encoding / scaling / PCA / KNN-fit pipeline. Returns everything the
    scouting function needs.
    """

    df = pd.read_csv(csv_path, low_memory=False)

    df_ml = df[[
        'player_positions', 'overall', 'potential', 'age', 'height_cm', 'weight_kg',
        'league_level', 'club_position',
        'preferred_foot', 'weak_foot', 'skill_moves', 'international_reputation',
        'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic',
        'attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy',
        'attacking_short_passing', 'attacking_volleys',
        'skill_dribbling', 'skill_curve', 'skill_fk_accuracy', 'skill_long_passing',
        'skill_ball_control',
        'movement_acceleration', 'movement_sprint_speed', 'movement_agility',
        'movement_reactions', 'movement_balance',
        'power_shot_power', 'power_jumping', 'power_stamina', 'power_strength',
        'power_long_shots',
        'mentality_aggression', 'mentality_interceptions', 'mentality_positioning',
        'mentality_vision', 'mentality_penalties', 'mentality_composure',
        'defending_marking_awareness', 'defending_standing_tackle',
        'defending_sliding_tackle',
    ]].copy()

    df_financial = df[['wage_eur', 'value_eur', 'club_contract_valid_until_year']]

    # ---- preprocessing ----
    df_ml = df_ml.dropna()

    # keep df, df_financial rows aligned with df_ml index (they already share the same index)

    # ---- feature engineering ----
    df_ml['bmi'] = df_ml['weight_kg'] / ((df_ml['height_cm'] / 100) ** 2)

    df_ml['speed_score'] = (
        df_ml['movement_acceleration'] + df_ml['movement_sprint_speed']
    ) / 2

    df_ml['speed_profile'] = (
        df_ml['movement_sprint_speed'] - df_ml['movement_acceleration']
    )

    df_ml['sub_shooting_score'] = (
        df_ml['attacking_finishing'] + df_ml['power_shot_power'] + df_ml['power_long_shots']
    ) / 3

    df_ml['sub_passing_score'] = (
        df_ml['attacking_short_passing'] + df_ml['skill_long_passing'] + df_ml['mentality_vision']
    ) / 3

    df_ml['sub_dribbling_score'] = (
        df_ml['skill_dribbling'] + df_ml['skill_ball_control'] + df_ml['movement_agility']
    ) / 3

    df_ml['sub_defending_score'] = (
        df_ml['defending_marking_awareness'] + df_ml['defending_standing_tackle']
        + df_ml['defending_sliding_tackle']
    ) / 3

    df_ml['mentality_score'] = (
        df_ml['mentality_aggression'] + df_ml['mentality_positioning']
        + df_ml['mentality_vision'] + df_ml['mentality_composure']
    ) / 4

    df_ml['growth_potential'] = df_ml['potential'] - df_ml['overall']
    df_ml['contract_remaining_years'] = (
        df_financial['club_contract_valid_until_year'] - 2026
    )

    df_ml['num_positions'] = (
        df_ml['player_positions'].fillna('')
        .apply(lambda x: len([p.strip() for p in x.split(',') if p.strip()]))
    )
    df_ml['primary_position'] = (
        df_ml['player_positions'].fillna('')
        .apply(lambda x: x.split(',')[0].strip() if x else 'Unknown')
    )

    cols_to_drop = ['club_contract_valid_until_year', 'player_positions']
    df_ml = df_ml.drop(columns=[c for c in cols_to_drop if c in df_ml.columns])

    # ---- selection ----
    meta_cols = ['short_name', 'value_eur', 'wage_eur', 'club_name', 'league_name', 'nationality_name']

    skill_cols = [
        'overall', 'potential', 'age', 'preferred_foot', 'weak_foot', 'skill_moves',
        'international_reputation', 'pace', 'shooting', 'passing', 'dribbling', 'defending',
        'physic', 'attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy',
        'attacking_short_passing', 'attacking_volleys', 'skill_dribbling', 'skill_curve',
        'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control',
        'movement_acceleration', 'movement_sprint_speed', 'movement_agility',
        'movement_reactions', 'movement_balance', 'power_shot_power', 'power_jumping',
        'power_stamina', 'power_strength', 'power_long_shots', 'mentality_aggression',
        'mentality_interceptions', 'mentality_positioning', 'mentality_vision',
        'mentality_penalties', 'mentality_composure', 'defending_marking_awareness',
        'defending_standing_tackle', 'defending_sliding_tackle', 'bmi', 'primary_position',
        'speed_score', 'speed_profile', 'sub_shooting_score', 'sub_passing_score',
        'sub_dribbling_score', 'sub_defending_score', 'contract_remaining_years',
        'mentality_score', 'growth_potential', 'num_positions',
    ]

    # merge meta cols (value_eur/wage_eur/club_name/league_name/nationality_name/short_name)
    # from the original df since df_ml doesn't carry them yet
    for c in meta_cols:
        if c not in df_ml.columns and c in df.columns:
            df_ml[c] = df.loc[df_ml.index, c]

    selected_cols = [col for col in meta_cols + skill_cols if col in df_ml.columns]
    df_ml = df_ml[selected_cols].copy()

    # ---- encoding ----
    df_encoded = pd.get_dummies(df_ml, columns=['primary_position', 'preferred_foot'], drop_first=False)

    exclude_from_similarity = [
        'short_name', 'league_name', 'club_name', 'nationality_name',
        'value_eur', 'wage_eur', 'release_clause_eur', 'age', 'overall',
    ]

    features_to_scale = [
        col for col in df_encoded.select_dtypes(include=[np.number]).columns
        if col not in exclude_from_similarity
    ]

    scaler = StandardScaler()
    df_ml_scaled = df_encoded.copy()
    df_ml_scaled[features_to_scale] = scaler.fit_transform(
        df_ml_scaled[features_to_scale].fillna(0)
    )

    # ---- PCA ----
    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(df_ml_scaled[features_to_scale].fillna(0))

    # ---- KNN engine ----
    knn_engine = NearestNeighbors(metric='euclidean')
    knn_engine.fit(X_pca)

    return {
        'df': df,
        'df_ml': df_ml,
        'X_pca': X_pca,
        'knn_engine': knn_engine,
        'features_to_scale': features_to_scale,
        'pca': pca,
        'scaler': scaler,
    }


def scout_custom_targets(
    ctx,
    player_name,
    max_age=24,
    max_budget_eur=None,
    top_n=5,
    same_pos_only=False,
):
    df = ctx['df']
    df_ml = ctx['df_ml']
    X_pca = ctx['X_pca']
    knn_engine = ctx['knn_engine']

    player_matches = df[df[name_col].str.contains(player_name, case=False, na=False)]

    if len(player_matches) == 0:
        return f"can not find a player with this name : {player_name}", None

    target_idx = player_matches.index[0]
    target_player_name = df.loc[target_idx, name_col]
    target_price = df.loc[target_idx].get('value_eur', 0)

    if target_idx not in df_ml.index:
        return (
            f"'{target_player_name}' was dropped during preprocessing "
            f"(missing values in required stats), so no comparison is possible.",
            None,
        )

    target_pos = df_ml.loc[target_idx].get('primary_position', '')

    if max_budget_eur is None:
        max_budget_eur = target_price

    loc_in_pca = df_ml.index.get_loc(target_idx)
    target_pca = X_pca[loc_in_pca].reshape(1, -1)

    n_neighbors = min(10000, X_pca.shape[0])
    distances, indices = knn_engine.kneighbors(target_pca, n_neighbors=n_neighbors)

    # Drop the target itself (distance 0 at position 0) before computing the
    # min/max range used to normalize similarity — this range is computed
    # once, over the FULL candidate pool, so it doesn't shift depending on
    # how many results end up passing the age/budget/position filters.
    pool_distances = distances[0][1:]
    min_dist_for_player = pool_distances.min()
    max_dist_for_player = pool_distances.max()

    results = []
    for dist, loc_idx in zip(distances[0][1:], indices[0][1:]):
        cand_idx = df_ml.index[loc_idx]

        cand_price = df.loc[cand_idx].get('value_eur', 0)
        cand_age = df.loc[cand_idx].get('age', 99)
        cand_pos = df_ml.loc[cand_idx].get('primary_position', '')

        if cand_age > max_age:
            continue
        if cand_price > max_budget_eur:
            continue
        if same_pos_only and cand_pos != target_pos:
            continue

        if max_dist_for_player == min_dist_for_player:
            sim_score = 100.0
        else:
            normalized_dist = (
                (dist - min_dist_for_player) / (max_dist_for_player - min_dist_for_player)
            )
            sim_score = (1 - normalized_dist) * 100
            sim_score = np.clip(sim_score, 0, 100)

        results.append({
            'Name': df.loc[cand_idx, name_col],
            'Position': df.loc[cand_idx].get('player_positions', cand_pos),
            'Similarity Score': f'{sim_score:.1f}%',
            'Age': cand_age,
            'Value (€)': f'{cand_price:,.0f}',
            'Overall': df.loc[cand_idx].get('overall', 'N/A'),
            'Potential': df.loc[cand_idx].get('potential', 'N/A'),
            'Club': df.loc[cand_idx].get('club_name', 'N/A'),
        })

        if len(results) == top_n:
            break

    if len(results) == 0:
        return (
            'No alternatives meeting these criteria were found; '
            'try increasing the budget or raising the age limit.',
            pd.DataFrame(),
        )

    header = (
        f"Alternatives for {target_player_name} "
        f"(max age {max_age}, max budget €{max_budget_eur:,.0f}):"
    )
    return header, pd.DataFrame(results)



# ============================================================
# Combined Streamlit application
# ============================================================
st.set_page_config(
    page_title="AI Football Scout | Player Scout",
    page_icon="⚽",
    layout="wide"
)

# الكود يتحط هنا بالظبط (بره وتحت)
st.markdown("""
    <style>
    * {
        direction: ltr !important;
    }
    </style>
""", unsafe_allow_html=True)

# وبعدين تكمل كودك العادي جداً
tab_classifier, tab_scout = st.tabs([
    "⚽ AI Football Scout: Position Classifier",
    "🔎 Player Scout — Find Similar / Cheaper Alternatives",
])

with tab_classifier:
    # ... باقي الكود ...
    # 1. إعدادات الصفحة
    # ==========================================

    st.title("⚽ AI Football Scout: Position Classifier")
    st.markdown("Enter the Detailed Status of the player, The ai will predict the bist position")

    # ==========================================
    # 2. تحميل الموديلات والـ Transformers
    # ==========================================
    @st.cache_resource
    def load_models():
        try:
            scaler = joblib.load('scaler.pkl')
            pca = joblib.load('pca_final.pkl')
            model = joblib.load('xgb_model.pkl')
            le = joblib.load('label_encoder.pkl')
            return scaler, pca, model, le
        except Exception as e:
            st.error("⚠️ Please make sure files (scaler.pkl, pca_final.pkl, xgb_model.pkl, label_encoder.pkl) are in path.")
            st.stop()

    scaler, pca_final, xgb_model, label_encoder = load_models()

    # ==========================================
    # 3. واجهة إدخال البيانات (UI)
    # ==========================================
    # تقسيم الشاشة لـ Tabs لتسهيل إدخال الـ 44 ميزة
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Basic Info", "⚡ Main Stats", "🎯 Attacking & Skill", "🏃 Movement & Power", "🛡️ Mentality & Defending"])

    input_data = {}

    with tab1:
        st.subheader("Basic Information")
        col1, col2, col3 = st.columns(3)
        input_data['age'] = col1.number_input("Age", 15, 45, 25)
        input_data['height_cm'] = col2.number_input("Height (cm)", 150, 220, 180)
        input_data['weight_kg'] = col3.number_input("Weight (kg)", 50, 110, 75)

        col4, col5, col6 = st.columns(3)
        input_data['overall'] = col4.slider("Overall", 40, 99, 75)
        input_data['potential'] = col5.slider("Potential", 40, 99, 80)
        input_data['international_reputation'] = col6.slider("Int. Reputation", 1, 5, 1)

        col7, col8 = st.columns(2)
        pref_foot = col7.selectbox("Preferred Foot", ["Left", "Right"])
        input_data['preferred_foot'] = 0 if pref_foot == "Left" else 1
        input_data['weak_foot'] = col8.slider("Weak Foot", 1, 5, 3)
        input_data['skill_moves'] = st.slider("Skill Moves", 1, 5, 3)

    with tab2:
        st.subheader("Main Attributes")
        col1, col2 = st.columns(2)
        input_data['pace'] = col1.slider("Pace", 1, 99, 70)
        input_data['shooting'] = col2.slider("Shooting", 1, 99, 70)
        input_data['passing'] = col1.slider("Passing", 1, 99, 70)
        input_data['dribbling'] = col2.slider("Dribbling", 1, 99, 70)
        input_data['defending'] = col1.slider("Defending", 1, 99, 70)
        input_data['physic'] = col2.slider("Physic", 1, 99, 70)

    with tab3:
        st.subheader("Attacking & Skill")
        col1, col2 = st.columns(2)
        input_data['attacking_crossing'] = col1.slider("Crossing", 1, 99, 50)
        input_data['attacking_finishing'] = col2.slider("Finishing", 1, 99, 50)
        input_data['attacking_heading_accuracy'] = col1.slider("Heading Accuracy", 1, 99, 50)
        input_data['attacking_short_passing'] = col2.slider("Short Passing", 1, 99, 50)
        input_data['attacking_volleys'] = col1.slider("Volleys", 1, 99, 50)

        input_data['skill_dribbling'] = col2.slider("Skill Dribbling", 1, 99, 50)
        input_data['skill_curve'] = col1.slider("Curve", 1, 99, 50)
        input_data['skill_fk_accuracy'] = col2.slider("FK Accuracy", 1, 99, 50)
        input_data['skill_long_passing'] = col1.slider("Long Passing", 1, 99, 50)
        input_data['skill_ball_control'] = col2.slider("Ball Control", 1, 99, 50)

    with tab4:
        st.subheader("Movement & Power")
        col1, col2 = st.columns(2)
        input_data['movement_acceleration'] = col1.slider("Acceleration", 1, 99, 50)
        input_data['movement_sprint_speed'] = col2.slider("Sprint Speed", 1, 99, 50)
        input_data['movement_agility'] = col1.slider("Agility", 1, 99, 50)
        input_data['movement_reactions'] = col2.slider("Reactions", 1, 99, 50)
        input_data['movement_balance'] = col1.slider("Balance", 1, 99, 50)

        input_data['power_shot_power'] = col2.slider("Shot Power", 1, 99, 50)
        input_data['power_jumping'] = col1.slider("Jumping", 1, 99, 50)
        input_data['power_stamina'] = col2.slider("Stamina", 1, 99, 50)
        input_data['power_strength'] = col1.slider("Strength", 1, 99, 50)
        input_data['power_long_shots'] = col2.slider("Long Shots", 1, 99, 50)

    with tab5:
        st.subheader("Mentality & Defending")
        col1, col2 = st.columns(2)
        input_data['mentality_aggression'] = col1.slider("Aggression", 1, 99, 50)
        input_data['mentality_interceptions'] = col2.slider("Interceptions", 1, 99, 50)
        input_data['mentality_positioning'] = col1.slider("Positioning", 1, 99, 50)
        input_data['mentality_vision'] = col2.slider("Vision", 1, 99, 50)
        input_data['mentality_penalties'] = col1.slider("Penalties", 1, 99, 50)
        input_data['mentality_composure'] = col2.slider("Composure", 1, 99, 50)

        input_data['defending_marking_awareness'] = col1.slider("Marking", 1, 99, 50)
        input_data['defending_standing_tackle'] = col2.slider("Standing Tackle", 1, 99, 50)
        input_data['defending_sliding_tackle'] = col1.slider("Sliding Tackle", 1, 99, 50)


    # ==========================================
    # 4. معالجة البيانات والتوقع (Pipeline)
    # ==========================================
    st.markdown("---")
    if st.button("🚀 Analyze & Predict Position", use_container_width=True):
        with st.spinner("Analyzing player attributes..."):

            # ترتيب العواميد بنفس الشكل الموجود في النوت بوك
            main_ml_columns = ['age', 'height_cm', 'weight_kg', 'overall', 'potential', 'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic', 'weak_foot', 'skill_moves', 'international_reputation', 'preferred_foot']
            sub_ml_columns = ['attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy', 'attacking_short_passing', 'attacking_volleys', 'skill_dribbling', 'skill_curve', 'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control', 'movement_acceleration', 'movement_sprint_speed', 'movement_agility', 'movement_reactions', 'movement_balance', 'power_shot_power', 'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots', 'mentality_aggression', 'mentality_interceptions', 'mentality_positioning', 'mentality_vision', 'mentality_penalties', 'mentality_composure', 'defending_marking_awareness', 'defending_standing_tackle', 'defending_sliding_tackle']

            features_to_scale = main_ml_columns + sub_ml_columns

            # تحويل بيانات المستخدم لـ DataFrame
            df_input = pd.DataFrame([input_data])[features_to_scale]

            # 1. StandardScaler لكل العواميد (44 عمود)
            scaled_array = scaler.transform(df_input)

            # 2. فصل العواميد الأساسية (أول 15) عن الفرعية (آخر 29)
            main_scaled = scaled_array[:, :15]
            sub_scaled = scaled_array[:, 15:]

            # 3. تطبيق PCA على العواميد الفرعية فقط
            sub_pca = pca_final.transform(sub_scaled)

            # 4. دمج العواميد الأساسية مع ناتج الـ PCA (السطر اللي كان ممسوح!)
            final_input_features = np.hstack((main_scaled, sub_pca))

            # 5. التوقع باستخدام XGBoost
            prediction_encoded = xgb_model.predict(final_input_features)

            # تحويل الرقم لـ Integer صريح لتجنب أي تعارض مع مكتبة sklearn
            pred_idx = int(prediction_encoded[0])

            # 6. تحويل التوقع لاسم المركز
            predicted_position = label_encoder.inverse_transform([pred_idx])[0]

            st.success("✅ Analysis Complete!")
            st.markdown(f"""
            <div style='background-color:#1E1E1E; padding:20px; border-radius:10px; text-align:center;'>
                <h3 style='color:#FFFFFF; margin-bottom: 0px;'>Optimal Playing Position</h3>
                <h1 style='color:#00FF00; font-size: 50px; margin-top: 10px;'>{predicted_position}</h1>
            </div>
            """, unsafe_allow_html=True)

with tab_scout:


    st.title("⚽ Player Scout — Find Similar / Cheaper Alternatives")
    st.caption(
        "Enter a player's name and find similar players (by playing style) "
        "who fit your age and budget constraints."
    )

    DEFAULT_CSV_PATH = "FC26_20250921.csv"


    @st.cache_resource(show_spinner="Loading & processing player data (first run only)...")
    def get_context(csv_source):
        return load_and_prepare(csv_source)


    # ---- Data source ----
    with st.sidebar:
        st.header("Data")
        uploaded = st.file_uploader("Upload FC dataset CSV", type=["csv"])
        st.caption(
            "If you don't upload a file, the app will look for "
            f"`{DEFAULT_CSV_PATH}` next to app.py."
        )

    csv_source = uploaded if uploaded is not None else DEFAULT_CSV_PATH

    try:
        ctx = get_context(csv_source)
    except FileNotFoundError:
        st.warning(
            f"Couldn't find `{DEFAULT_CSV_PATH}`. Please upload the dataset CSV "
            "from the sidebar to get started."
        )
        st.stop()
    except Exception as e:
        st.error(f"Failed to load/process the dataset: {e}")
        st.stop()

    st.success(f"Data loaded — {ctx['df_ml'].shape[0]:,} players available for comparison.")

    st.divider()

    # ---- Search form ----
    col1, col2 = st.columns([2, 1])

    with col1:
        player_name = st.text_input("Player name", placeholder="e.g. Bellingham")

    with col2:
        top_n = st.number_input("Number of results", min_value=1, max_value=50, value=5, step=1)

    col3, col4, col5 = st.columns(3)

    with col3:
        max_age = st.slider("Max age", min_value=15, max_value=45, value=24)

    with col4:
        use_custom_budget = st.checkbox("Set custom max budget (€)", value=False)
        max_budget = None
        if use_custom_budget:
            max_budget = st.number_input(
                "Max budget (€)", min_value=0, value=10_000_000, step=100_000, format="%d"
            )

    with col5:
        same_pos_only = st.checkbox("Same primary position only", value=False)

    search_clicked = st.button("🔍 Search", type="primary", use_container_width=True)

    st.divider()

    if search_clicked:
        if not player_name.strip():
            st.warning("Please enter a player name.")
        else:
            with st.spinner("Searching..."):
                message, results_df = scout_custom_targets(
                    ctx,
                    player_name.strip(),
                    max_age=max_age,
                    max_budget_eur=max_budget,
                    top_n=int(top_n),
                    same_pos_only=same_pos_only,
                )

            if results_df is None:
                st.error(message)
            elif results_df.empty:
                st.info(message)
            else:
                st.subheader(message)
                st.dataframe(results_df, use_container_width=True, hide_index=True)

                csv_bytes = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download results as CSV",
                    data=csv_bytes,
                    file_name=f"scout_results_{player_name.strip()}.csv",
                    mime="text/csv",
                )
    else:
        st.info("Enter a player name above and press **Search** to get started.")
