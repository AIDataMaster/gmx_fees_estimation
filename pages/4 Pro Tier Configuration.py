# %% Imports
import streamlit as st
import pandas as pd
import os

from functions.functions import (
    centered_header,
    get_default_pro_tiers_sum,
    get_default_pro_tiers_or
)

# %% Page config
st.set_page_config(page_title="Pro Tier Configuration", layout="wide", page_icon=":bar_chart:")

# %% Header
centered_header("Pro Tier Configuration")

st.markdown("""
    ---
    This page allows you to **define and customize Pro Tier settings** based on:

    - Volume thresholds  
    - Staking thresholds  
    - Discounts for each rule (volume, staking, both)  
    - Condition logic (**OR**, AND, SUM, BOTH)

    💡 **These settings will be used in the _Fee Estimation_ page for all downstream calculations.**
""")

# %% Versioning
CONFIG_FOLDER = "./config_versions"
os.makedirs(CONFIG_FOLDER, exist_ok=True)
available_versions = [f.replace(".csv", "") for f in os.listdir(CONFIG_FOLDER) if f.endswith(".csv")]

# %% Scenario 1
st.markdown("---")
st.header("🔵 Scenario 1 Configuration")

logic_options = ["SUM", "OR"]
default_logic = st.session_state.get("logic_applied_scenario_1", "SUM")
logic_index = logic_options.index(default_logic)
logic_scenario_1 = st.selectbox("Select Condition Logic for Scenario 1", options=logic_options, index=logic_index, key="logic_selector_1")

def get_default_table(logic):
    return get_default_pro_tiers_sum() if logic == "SUM" else get_default_pro_tiers_or()

# Track and reset if logic changed
if "previous_logic_scenario_1" not in st.session_state:
    st.session_state["previous_logic_scenario_1"] = logic_scenario_1
elif st.session_state["previous_logic_scenario_1"] != logic_scenario_1:
    st.session_state["pro_tiers_temp_scenario_1"] = get_default_table(logic_scenario_1)
    st.session_state["pro_tiers_selected_version_scenario_1"] = "<Select>"
    st.session_state["pro_tiers_last_loaded_scenario_1"] = None
    st.session_state["previous_logic_scenario_1"] = logic_scenario_1
    st.info(f"🔄 Logic changed to `{logic_scenario_1}` — table reset to default.")

if "pro_tiers_temp_scenario_1" not in st.session_state:
    st.session_state["pro_tiers_temp_scenario_1"] = get_default_table(logic_scenario_1)

if "pro_tiers_selected_version_scenario_1" not in st.session_state:
    st.session_state["pro_tiers_selected_version_scenario_1"] = "<Select>"

if "pro_tiers_applied_version_scenario_1" in st.session_state:
    applied = st.session_state["pro_tiers_applied_version_scenario_1"]
    st.info(f"✅ **Using config:** `{applied}` for Fee Estimation (Scenario 1)")
else:
    st.warning("⚠️ **No config has been applied yet.** Click 'Apply for Fee Estimation (Scenario 1)' to activate.")

col1, col2, col3, col4 = st.columns([1.2, 2, 2, 2])

with col1:
    if st.button("🔁 Reset to Default", key="reset_s1"):
        st.session_state["pro_tiers_temp_scenario_1"] = get_default_table(logic_scenario_1)
        st.session_state["pro_tiers_selected_version_scenario_1"] = "<Select>"
        st.session_state["pro_tiers_last_loaded_scenario_1"] = None
        st.success("🔄 Reset to default config.")

with col2:
    dropdown_options = ["<Select>"] + available_versions
    current_selection = st.session_state["pro_tiers_selected_version_scenario_1"]
    try:
        selected_index = dropdown_options.index(current_selection)
    except ValueError:
        selected_index = 0

    selected_config = st.selectbox(
        "📂 Load Config Version",
        options=dropdown_options,
        index=selected_index,
        key="pro_tiers_selected_version_input_s1"
    )

    if selected_config != "<Select>" and selected_config != st.session_state.get("pro_tiers_last_loaded_scenario_1"):
        version_path = os.path.join(CONFIG_FOLDER, f"{selected_config}.csv")
        try:
            loaded_df = pd.read_csv(version_path)
            required_cols = get_default_table(logic_scenario_1).columns.tolist()
            missing = [col for col in required_cols if col not in loaded_df.columns]
            if missing:
                st.error(f"❌ Missing columns in uploaded config: {missing}")
            else:
                st.session_state["pro_tiers_temp_scenario_1"] = loaded_df[required_cols]
                st.session_state["pro_tiers_last_loaded_scenario_1"] = selected_config
                st.session_state["pro_tiers_selected_version_scenario_1"] = selected_config
                st.success(f"✅ Loaded config `{selected_config}`")
        except Exception as e:
            st.error(f"⚠️ Error loading config: {e}")

with col3:
    if st.button("✅ Apply for Fee Estimation (Scenario 1)", key="apply_s1"):
        st.session_state["pro_tiers_config_scenario_1"] = st.session_state["pro_tiers_temp_scenario_1"].copy()
        current_ver = st.session_state.get("pro_tiers_selected_version_scenario_1", "<Select>")
        if current_ver != "<Select>":
            st.session_state["pro_tiers_applied_version_scenario_1"] = current_ver
        else:
            st.session_state["pro_tiers_applied_version_scenario_1"] = "(unsaved config)"
        st.session_state["logic_applied_scenario_1"] = logic_scenario_1  # ✅ store logic explicitly
        st.success(f"✅ Config applied to Fee Estimation! Now using `{st.session_state['pro_tiers_applied_version_scenario_1']}`.")

with col4:
    config_name = st.text_input("💾 Save Config As", value="", placeholder="Enter config name", key="save_input_s1")
    if st.button("📁 Save to Disk", key="save_btn_s1"):
        if config_name.strip():
            save_path = os.path.join(CONFIG_FOLDER, f"{config_name}.csv")
            st.session_state["pro_tiers_temp_scenario_1"].to_csv(save_path, index=False)
            st.success(f"💾 Saved as `{config_name}.csv`")
        else:
            st.warning("⚠️ Please enter a valid name for saving.")

# Upload
st.markdown("### 📤 Upload Config File (CSV)")
uploaded_file = st.file_uploader("Upload a Pro Tier config CSV", type="csv", key="uploader_s1")

required_columns = get_default_table(logic_scenario_1).columns.tolist()

if uploaded_file:
    try:
        uploaded_df = pd.read_csv(uploaded_file)
        missing_cols = [col for col in required_columns if col not in uploaded_df.columns]
        if missing_cols:
            st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
        else:
            for col in uploaded_df.columns:
                uploaded_df[col] = pd.to_numeric(uploaded_df[col], errors='ignore')
            st.session_state["pro_tiers_temp_scenario_1"] = uploaded_df[required_columns]
            st.success("✅ Uploaded config loaded into session.")
    except Exception as e:
        st.error(f"❌ Could not read file: {e}")

# Edit Table
st.markdown("### ✏️ Edit Pro Tier Rules")

if logic_scenario_1 == "SUM":
    column_config = {
        "tier": st.column_config.TextColumn("Tier"),
        "volume_min": st.column_config.NumberColumn("Volume Min ($)", format="$,"),
        "staking_min": st.column_config.NumberColumn("Staking Min (GMX)", format=","),

        "discount_volume": st.column_config.NumberColumn("Discount for Volume", format="%.2f"),
        "discount_staking": st.column_config.NumberColumn("Discount for Staking", format="%.2f"),
    }
else:
    column_config = {
        "tier": st.column_config.TextColumn("Tier"),
        "volume_min": st.column_config.NumberColumn("Volume Min ($)", format="$,"),
        "staking_min": st.column_config.NumberColumn("Staking Min (GMX)", format=","),

        "discount_bps_pos": st.column_config.NumberColumn("Positive BPS Discount", format="%.6f"),
        "discount_bps_neg": st.column_config.NumberColumn("Negative BPS Discount", format="%.6f"),
        "discount_both_bps_pos": st.column_config.NumberColumn("Both Positive BPS", format="%.6f"),
        "discount_both_bps_neg": st.column_config.NumberColumn("Both Negative BPS", format="%.6f"),
    }


edited_df = st.data_editor(
    st.session_state["pro_tiers_temp_scenario_1"],
    num_rows="dynamic",
    column_config=column_config
)

if st.button("💾 Save Changes to Table", key="save_temp_s1"):
    cleaned_df = edited_df.dropna(how="all").copy()
    st.session_state["pro_tiers_temp_scenario_1"] = cleaned_df
    st.success("✅ Changes saved to session (not applied until Apply is clicked).")

# Final preview
st.markdown("### ✅ Final Pro Tier Table")

df = st.session_state["pro_tiers_temp_scenario_1"].dropna(how="all").copy()

formatting = {
    "volume_min": "${:,.0f}",
    "staking_min": "{:,.0f}",
}

if logic_scenario_1 == "SUM":
    formatting.update({
        "discount_volume": "{:.2%}",
        "discount_staking": "{:.2%}",
    })
else:
    # Step 1 — Calculate discount percentages
    df["discount_pos"] = 1 - (df["discount_bps_pos"] / 0.0004)
    df["discount_neg"] = 1 - (df["discount_bps_neg"] / 0.0006)
    df["discount_both_pos"] = 1 - (df["discount_both_bps_pos"] / 0.0004)
    df["discount_both_neg"] = 1 - (df["discount_both_bps_neg"] / 0.0006)
    
    # Step 2 - Format outputs
    formatting.update({
        "discount_bps_pos": "{:.6f}",
        "discount_bps_neg": "{:.6f}",
        "discount_both_bps_pos": "{:.6f}",
        "discount_both_bps_neg": "{:.6f}",
        "discount_pos": "{:.2%}",
        "discount_neg": "{:.2%}",
        "discount_both_pos": "{:.2%}",
        "discount_both_neg": "{:.2%}",
    })

st.dataframe(df.style.format(formatting), use_container_width=True)

# %% Scenario 2 Configuration
st.markdown("---")
st.header("🟢 Scenario 2 Configuration")

# Select logic for Scenario 2
logic_options = ["SUM", "OR"]
def_logic = st.session_state.get("logic_applied_scenario_2", "OR")
logic_index = logic_options.index(def_logic)
logic_scenario_2 = st.selectbox("Select Condition Logic for Scenario 2", options=logic_options, index=logic_index, key="logic_selector_2")

def get_default_table_s2(logic):
    return get_default_pro_tiers_sum() if logic == "SUM" else get_default_pro_tiers_or()

# Handle logic switching
if "previous_logic_scenario_2" not in st.session_state:
    st.session_state["previous_logic_scenario_2"] = logic_scenario_2
elif st.session_state["previous_logic_scenario_2"] != logic_scenario_2:
    st.session_state["pro_tiers_temp_scenario_2"] = get_default_table_s2(logic_scenario_2)
    st.session_state["pro_tiers_selected_version_scenario_2"] = "<Select>"
    st.session_state["pro_tiers_last_loaded_scenario_2"] = None
    st.session_state["previous_logic_scenario_2"] = logic_scenario_2
    st.info(f"🔄 Logic changed to `{logic_scenario_2}` — table reset to default.")

# Init state
if "pro_tiers_temp_scenario_2" not in st.session_state:
    st.session_state["pro_tiers_temp_scenario_2"] = get_default_table_s2(logic_scenario_2)

if "pro_tiers_selected_version_scenario_2" not in st.session_state:
    st.session_state["pro_tiers_selected_version_scenario_2"] = "<Select>"

# Display current config info
if "pro_tiers_applied_version_scenario_2" in st.session_state:
    applied = st.session_state["pro_tiers_applied_version_scenario_2"]
    st.info(f"✅ **Using config:** `{applied}` for Fee Estimation (Scenario 2)")
else:
    st.warning("⚠️ **No config has been applied yet.** Click 'Apply for Fee Estimation (Scenario 2)' to activate.")

# Buttons: Reset / Load / Apply / Save
col1, col2, col3, col4 = st.columns([1.2, 2, 2, 2])

with col1:
    if st.button("🔁 Reset to Default", key="reset_s2"):
        st.session_state["pro_tiers_temp_scenario_2"] = get_default_table_s2(logic_scenario_2)
        st.session_state["pro_tiers_selected_version_scenario_2"] = "<Select>"
        st.session_state["pro_tiers_last_loaded_scenario_2"] = None
        st.success("🔄 Reset to default config.")

with col2:
    dropdown_options = ["<Select>"] + available_versions
    current_selection = st.session_state["pro_tiers_selected_version_scenario_2"]
    try:
        selected_index = dropdown_options.index(current_selection)
    except ValueError:
        selected_index = 0

    selected_config = st.selectbox(
        "📂 Load Config Version",
        options=dropdown_options,
        index=selected_index,
        key="pro_tiers_selected_version_input_s2"
    )

    if selected_config != "<Select>" and selected_config != st.session_state.get("pro_tiers_last_loaded_scenario_2"):
        version_path = os.path.join(CONFIG_FOLDER, f"{selected_config}.csv")
        try:
            loaded_df = pd.read_csv(version_path)
            required_cols = get_default_table_s2(logic_scenario_2).columns.tolist()
            missing = [col for col in required_cols if col not in loaded_df.columns]
            if missing:
                st.error(f"❌ Missing columns in uploaded config: {missing}")
            else:
                st.session_state["pro_tiers_temp_scenario_2"] = loaded_df[required_cols]
                st.session_state["pro_tiers_last_loaded_scenario_2"] = selected_config
                st.session_state["pro_tiers_selected_version_scenario_2"] = selected_config
                st.success(f"✅ Loaded config `{selected_config}`")
        except Exception as e:
            st.error(f"⚠️ Error loading config: {e}")

with col3:
    if st.button("✅ Apply for Fee Estimation (Scenario 2)", key="apply_s2"):
        st.session_state["pro_tiers_config_scenario_2"] = st.session_state["pro_tiers_temp_scenario_2"].copy()
        current_ver = st.session_state.get("pro_tiers_selected_version_scenario_2", "<Select>")
        if current_ver != "<Select>":
            st.session_state["pro_tiers_applied_version_scenario_2"] = current_ver
        else:
            st.session_state["pro_tiers_applied_version_scenario_2"] = "(unsaved config)"
        st.session_state["logic_applied_scenario_2"] = logic_scenario_2  # ✅ store logic explicitly
        st.success(f"✅ Config applied to Fee Estimation! Now using `{st.session_state['pro_tiers_applied_version_scenario_2']}`.")

with col4:
    config_name = st.text_input("💾 Save Config As", value="", placeholder="Enter config name", key="save_input_s2")
    if st.button("📁 Save to Disk", key="save_btn_s2"):
        if config_name.strip():
            save_path = os.path.join(CONFIG_FOLDER, f"{config_name}.csv")
            st.session_state["pro_tiers_temp_scenario_2"].to_csv(save_path, index=False)
            st.success(f"💾 Saved as `{config_name}.csv`")
        else:
            st.warning("⚠️ Please enter a valid name for saving.")

# Upload CSV
st.markdown("### 📤 Upload Config File (CSV)")
uploaded_file = st.file_uploader("Upload a Pro Tier config CSV", type="csv", key="uploader_s2")

required_columns = get_default_table_s2(logic_scenario_2).columns.tolist()

if uploaded_file:
    try:
        uploaded_df = pd.read_csv(uploaded_file)
        missing_cols = [col for col in required_columns if col not in uploaded_df.columns]
        if missing_cols:
            st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
        else:
            for col in uploaded_df.columns:
                uploaded_df[col] = pd.to_numeric(uploaded_df[col], errors='ignore')
            st.session_state["pro_tiers_temp_scenario_2"] = uploaded_df[required_columns]
            st.success("✅ Uploaded config loaded into session.")
    except Exception as e:
        st.error(f"❌ Could not read file: {e}")

# Edit table
st.markdown("### ✏️ Edit Pro Tier Rules")

if logic_scenario_2 == "SUM":
    column_config = {
        "tier": st.column_config.TextColumn("Tier"),
        "volume_min": st.column_config.NumberColumn("Volume Min ($)", format="$,"),
        "staking_min": st.column_config.NumberColumn("Staking Min (GMX)", format=","),
        "discount_volume": st.column_config.NumberColumn("Discount for Volume", format="%.2f"),
        "discount_staking": st.column_config.NumberColumn("Discount for Staking", format="%.2f"),
    }
else:
    column_config = {
        "tier": st.column_config.TextColumn("Tier"),
        "volume_min": st.column_config.NumberColumn("Volume Min ($)", format="$,"),
        "staking_min": st.column_config.NumberColumn("Staking Min (GMX)", format=","),
        "discount_bps_pos": st.column_config.NumberColumn("Positive BPS Discount", format="%.6f"),
        "discount_bps_neg": st.column_config.NumberColumn("Negative BPS Discount", format="%.6f"),
        "discount_both_bps_pos": st.column_config.NumberColumn("Both Positive BPS", format="%.6f"),
        "discount_both_bps_neg": st.column_config.NumberColumn("Both Negative BPS", format="%.6f"),
    }

edited_df = st.data_editor(
    st.session_state["pro_tiers_temp_scenario_2"],
    key="data_editor_s2",
    num_rows="dynamic",
    column_config=column_config
)

if st.button("💾 Save Changes to Table", key="save_temp_s2"):
    cleaned_df = edited_df.dropna(how="all").copy()
    st.session_state["pro_tiers_temp_scenario_2"] = cleaned_df
    st.success("✅ Changes saved to session (not applied until Apply is clicked).")

# Preview
st.markdown("### ✅ Final Pro Tier Table")

df2 = st.session_state["pro_tiers_temp_scenario_2"].dropna(how="all").copy()

formatting_2  = {
    "volume_min": "${:,.0f}",
    "staking_min": "{:,.0f}",
}

if logic_scenario_2 == "SUM":
    formatting_2 .update({
        "discount_volume": "{:.2%}",
        "discount_staking": "{:.2%}",
    })
else:
    # Step 1 — Calculate discount % vs baseline BPS
    df2["discount_pos"] = 1 - (df2["discount_bps_pos"] / 0.0004)
    df2["discount_neg"] = 1 - (df2["discount_bps_neg"] / 0.0006)
    df2["discount_both_pos"] = 1 - (df2["discount_both_bps_pos"] / 0.0004)
    df2["discount_both_neg"] = 1 - (df2["discount_both_bps_neg"] / 0.0006)

    # Step 2 — Formatting
    formatting_2.update({
        "discount_bps_pos": "{:.6f}",
        "discount_bps_neg": "{:.6f}",
        "discount_both_bps_pos": "{:.6f}",
        "discount_both_bps_neg": "{:.6f}",
        "discount_pos": "{:.2%}",
        "discount_neg": "{:.2%}",
        "discount_both_pos": "{:.2%}",
        "discount_both_neg": "{:.2%}",
    })

st.dataframe(df2.style.format(formatting_2), use_container_width=True)

