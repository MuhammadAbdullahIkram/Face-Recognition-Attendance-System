"""Streamlit UI for the Face Recognition & Attendance System.
Run:  streamlit run app.py
"""
import cv2
import numpy as np
import streamlit as st

from face_engine import FaceEngine, load_attendance, mark_attendance

st.set_page_config(page_title="Face Attendance System", page_icon="🧑‍💼", layout="wide")


@st.cache_resource
def get_engine():
    return FaceEngine()


engine = get_engine()


def to_bgr(file):
    return cv2.imdecode(np.frombuffer(file.getvalue(), np.uint8), cv2.IMREAD_COLOR)


def show(bgr):
    st.image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


st.title("🧑‍💼 Face Recognition & Attendance System")
page = st.sidebar.radio("Menu", ["📋 Take Attendance", "➕ Register Person",
                                 "📊 Attendance Records", "⚙️ Manage"])
engine.threshold = st.sidebar.slider(
    "Match threshold (cosine)", 0.20, 0.60, float(engine.threshold), 0.01,
    help="Higher = stricter. Default 0.363 is the recommended SFace value.")
st.sidebar.caption(f"Registered people: {len(engine.people())}")

# ------------------------------------------------------------------ attendance
if page == "📋 Take Attendance":
    st.subheader("Mark attendance")
    if not engine.db:
        st.warning("No one is registered yet. Go to 'Register Person' first.")
    src = st.radio("Input", ["Camera", "Upload photo"], horizontal=True)
    file = st.camera_input("Look at the camera") if src == "Camera" else \
        st.file_uploader("Upload a photo (group photos work too)", type=["jpg", "jpeg", "png"])
    if file:
        bgr = to_bgr(file)
        results = engine.recognize(bgr)
        show(engine.annotate(bgr, results))
        if not results:
            st.warning("No face detected. Try better lighting / a clearer photo.")
        for r in results:
            if r["name"] == "Unknown":
                st.error(f"Unknown face (best score {r['score']:.2f})")
            elif mark_attendance(r["name"]):
                st.success(f"✅ {r['name']} marked present (score {r['score']:.2f})")
            else:
                st.info(f"ℹ️ {r['name']} is already marked today")

# ------------------------------------------------------------------- register
elif page == "➕ Register Person":
    st.subheader("Register a new person")
    name = st.text_input("Full name").strip()
    st.caption("Tip: 3-5 clear photos (different angles / lighting) give the best accuracy.")
    files = st.file_uploader("Upload photos", type=["jpg", "jpeg", "png"],
                             accept_multiple_files=True)
    cam = st.camera_input("...or take a photo")
    if st.button("Register", type="primary"):
        imgs = [to_bgr(f) for f in files] + ([to_bgr(cam)] if cam else [])
        if not name or not imgs:
            st.error("Enter a name and provide at least one photo.")
        else:
            ok = sum(engine.register(name, im) for im in imgs)
            if ok:
                st.success(f"Registered {ok}/{len(imgs)} photo(s) for {name}.")
            else:
                st.error("No face found in the provided photo(s).")

# -------------------------------------------------------------------- records
elif page == "📊 Attendance Records":
    st.subheader("Attendance records")
    df = load_attendance()
    if df.empty:
        st.info("No attendance marked yet.")
    else:
        dates = ["All"] + sorted(df["Date"].unique(), reverse=True)
        pick = st.selectbox("Date", dates)
        view = df if pick == "All" else df[df["Date"] == pick]
        c1, c2 = st.columns(2)
        c1.metric("Records", len(view))
        c2.metric("Unique people", view["Name"].nunique())
        st.dataframe(view, use_container_width=True)
        st.bar_chart(view.groupby("Name").size())
        st.download_button("⬇️ Download CSV", view.to_csv(index=False),
                           "attendance.csv", "text/csv")

# --------------------------------------------------------------------- manage
else:
    st.subheader("Registered people")
    people = engine.people()
    if not people:
        st.info("Nobody registered.")
    else:
        st.table({"Name": list(people), "Photos registered": list(people.values())})
        who = st.selectbox("Delete a person", list(people))
        if st.button("🗑️ Delete"):
            engine.delete(who)
            st.success(f"Deleted {who}.")
            st.rerun()
