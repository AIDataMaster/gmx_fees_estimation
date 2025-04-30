# %% Imports
import streamlit as st
import pandas as pd
import os

from functions.functions import centered_header

# %% Page config
st.set_page_config(page_title="Pro Tier Configuration", layout="wide", page_icon=":bar_chart:")

# %% Header
centered_header("Pro Tier Configuration")

st.markdown("""
This page allows you to **define and customize Pro Tier settings** based on:

- Volume thresholds  
- Staking thresholds  
- Discounts for each rule (volume, staking, both)  
- Condition logic (**OR**, AND, SUM, BOTH)

💡 **These settings will be used in the _Fee Estimation_ page for all downstream calculations.**
""")

# %% Default table
default_pro_tiers = pd.DataFrame({
    "tier": ["Pro 1", "Pro 2", "Pro 3", "Pro 4"],
    "volume_min": [0, 5_000_000, 40_000_000, 200_000_000],
    "volume_max": [5_000_000, 40_000_000, 200_000_000, None],
    "staking_min": [0, 20_000, 50_000, 100_000],
    "discount_volume": [0.00, 0.10, 0.20, 0.30],
    "discount_staking": [0.00, 0.10, 0.20, 0.30],
    "discount_both": [0.00, 0.15, 0.25, 0.35],
    "condition": ["OR", "OR", "OR", "OR"]
})

# %% Versioning
CONFIG_FOLDER = "./config_versions"
os.makedirs(CONFIG_FOLDER, exist_ok=True)
available_versions = [f.replace(".csv", "") for f in os.listdir(CONFIG_FOLDER) if f.endswith(".csv")]

# %% Initialize session state
if "pro_tiers_temp" not in st.session_state:
    st.session_state["pro_tiers_temp"] = default_pro_tiers.copy()
if "pro_tiers_selected_version" not in st.session_state:
    st.session_state["pro_tiers_selected_version"] = "<Select>"

# %% Show current applied config
if "pro_tiers_applied_version" in st.session_state:
    applied = st.session_state["pro_tiers_applied_version"]
    st.info(f"✅ **Using config:** `{applied}` for Fee Estimation")
else:
    st.warning("⚠️ **No config has been applied yet.** Click 'Apply for Fee Estimation' to activate.")

# %% Reset / Load / Save / Apply Controls
col1, col2, col3, col4 = st.columns([1.2, 2, 2, 2])

with col1:
    if st.button("🔁 Reset to Default"):
        st.session_state["pro_tiers_temp"] = default_pro_tiers.copy()
        st.session_state["pro_tiers_selected_version"] = "<Select>"
        st.session_state["pro_tiers_last_loaded"] = None
        st.success("🔄 Reset to default config.")

with col2:
    dropdown_options = ["<Select>"] + available_versions
    current_selection = st.session_state["pro_tiers_selected_version"]
    try:
        selected_index = dropdown_options.index(current_selection)
    except ValueError:
        selected_index = 0

    selected_config = st.selectbox(
        "📂 Load Config Version",
        options=dropdown_options,
        index=selected_index,
        key="pro_tiers_selected_version_input"
    )

    if selected_config != "<Select>" and selected_config != st.session_state.get("pro_tiers_last_loaded"):
        version_path = os.path.join(CONFIG_FOLDER, f"{selected_config}.csv")
        try:
            loaded_df = pd.read_csv(version_path)
            required_cols = default_pro_tiers.columns.tolist()
            missing = [col for col in required_cols if col not in loaded_df.columns]
            if missing:
                st.error(f"❌ Missing columns in uploaded config: {missing}")
            else:
                for col in required_cols:
                    if col not in loaded_df.columns:
                        loaded_df[col] = default_pro_tiers[col].dtype()
                loaded_df["condition"] = loaded_df["condition"].fillna("OR")
                st.session_state["pro_tiers_temp"] = loaded_df[required_cols]
                st.session_state["pro_tiers_last_loaded"] = selected_config
                st.session_state["pro_tiers_selected_version"] = selected_config
                st.success(f"✅ Loaded config `{selected_config}`")
        except Exception as e:
            st.error(f"⚠️ Error loading config: {e}")

with col3:
    if st.button("✅ Apply for Fee Estimation"):
        st.session_state["pro_tiers_config"] = st.session_state["pro_tiers_temp"].copy()
        current_ver = st.session_state.get("pro_tiers_selected_version", "<Select>")
        if current_ver != "<Select>":
            st.session_state["pro_tiers_applied_version"] = current_ver
        else:
            st.session_state["pro_tiers_applied_version"] = "(unsaved config)"
        st.success(f"✅ Config applied to Fee Estimation! Now using `{st.session_state['pro_tiers_applied_version']}`.")

with col4:
    config_name = st.text_input("💾 Save Config As", value="", placeholder="Enter config name")
    if st.button("📁 Save to Disk"):
        if config_name.strip():
            save_path = os.path.join(CONFIG_FOLDER, f"{config_name}.csv")
            st.session_state["pro_tiers_temp"].to_csv(save_path, index=False)
            st.success(f"💾 Saved as `{config_name}.csv`")
        else:
            st.warning("⚠️ Please enter a valid name for saving.")

# %% Upload
st.markdown("### 📤 Upload Config File (CSV)")
uploaded_file = st.file_uploader("Upload a Pro Tier config CSV", type="csv")

required_columns = default_pro_tiers.columns.tolist()

if uploaded_file:
    try:
        uploaded_df = pd.read_csv(uploaded_file)

        # Validate column presence
        missing_cols = [col for col in required_columns if col not in uploaded_df.columns]
        if missing_cols:
            st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
        else:
            # Validate types (simplified to numeric check)
            for col in ["volume_min", "volume_max", "staking_min", "discount_volume", "discount_staking", "discount_both"]:
                try:
                    uploaded_df[col] = pd.to_numeric(uploaded_df[col])
                except Exception:
                    st.error(f"❌ Column `{col}` must be numeric.")
                    st.stop()

            uploaded_df["condition"] = uploaded_df.get("condition", "OR").fillna("OR").replace("", "OR")

            st.session_state["pro_tiers_temp"] = uploaded_df[required_columns]
            st.success("✅ Uploaded config loaded into session.")
    except Exception as e:
        st.error(f"❌ Could not read file: {e}")

# %% Editable Data Table
st.markdown("### ✏️ Edit Pro Tier Rules")

edited_df = st.data_editor(
    st.session_state["pro_tiers_temp"],
    num_rows="dynamic",
    column_config={
        "volume_min": st.column_config.NumberColumn("Volume Min ($)", format="$,"),
        "volume_max": st.column_config.NumberColumn("Volume Max ($)", format="$,"),
        "staking_min": st.column_config.NumberColumn("Staking Min (GMX)", format=","),
        "discount_volume": st.column_config.NumberColumn("Discount for Volume", format="%.2f"),
        "discount_staking": st.column_config.NumberColumn("Discount for Staking", format="%.2f"),
        "discount_both": st.column_config.NumberColumn("Bonus if Both", format="%.2f"),
        "condition": st.column_config.SelectboxColumn("Condition Logic", options=["OR"]), # , "AND", "SUM", "BOTH"
    }
)

# Save temp changes button
if st.button("💾 Save Changes to Temp"):
    cleaned_df = edited_df.dropna(how="all").copy()
    cleaned_df["condition"] = cleaned_df["condition"].fillna("OR")
    st.session_state["pro_tiers_temp"] = cleaned_df
    st.success("✅ Changes saved to session (not applied until Apply is clicked).")

# %% Final Preview
st.markdown("### ✅ Final Pro Tier Table")
st.dataframe(
    st.session_state["pro_tiers_temp"].dropna(how="all").copy().fillna({"condition": "OR"}).style.format({
        "volume_min": "${:,.0f}",
        "volume_max": "${:,.0f}",
        "staking_min": "{:,.0f}",
        "discount_volume": "{:.2%}",
        "discount_staking": "{:.2%}",
        "discount_both": "{:.2%}",
    }),
    use_container_width=True
)
