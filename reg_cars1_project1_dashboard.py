import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

#-----sidebar-------
st.sidebar.image(r'C:\Users\omar\Desktop\reg_cars_streamlit\imge\cars.png')

#-----header-------
st.image(r"C:\Users\omar\Desktop\reg_cars_streamlit\imge\cars2.png")

st.sidebar.title("🧮 إدخال البيانات")

# ثابت: قائمة الماركات


brands_list = ['Maruti', 'Hyundai', 'Toyota', 'Ford', 'Honda', 'BMW', 'Mercedes', 'Audi', 'Nissan', 'Volkswagen']
brand = st.sidebar.selectbox("🏷️ اختر الماركة", brands_list, key="brand")

year = st.sidebar.number_input("📅 سنة الصنع", min_value=1990, max_value=2025, value=2015, key="year")
km_driven = st.sidebar.number_input("🚗 عدد الكيلومترات المقطوعة", min_value=0, value=50000, step=1000, key="km_driven")
fuel = st.sidebar.selectbox("⛽ نوع الوقود", ["Petrol", "Diesel", "CNG", "LPG", "Electric"], key="fuel")
seller_type = st.sidebar.selectbox("🧍 نوع البائع", ["Individual", "Dealer", "Trustmark Dealer"], key="seller_type")
transmission = st.sidebar.selectbox("⚙️ ناقل الحركة", ["Manual", "Automatic"], key="transmission")
owner = st.sidebar.selectbox("👥 عدد المالكين السابقين", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"], key="owner")
mileage = st.sidebar.number_input("📏 المسافة المقطوعة (كم/لتر)", min_value=0.0, value=18.0, key="mileage")
engine = st.sidebar.number_input("🧠 سعة المحرك (cc)", min_value=500, max_value=6000, value=1500, key="engine")
max_power = st.sidebar.number_input("⚡ أقصى قوة (حصان)", min_value=30.0, value=100.0, key="max_power")
seats = st.sidebar.selectbox("💺 عدد المقاعد", [2, 4, 5, 6, 7, 8, 9, 10], key="seats")
torque = st.sidebar.number_input("🌀 عزم الدوران (Nm)", min_value=20.0, max_value=1000.0, value=113.8, key="torque")
# تعليمات بجانب الحقول
st.sidebar.markdown("### تعليمات الإدخال:")
st.sidebar.markdown("""
- اختر الماركة من القائمة المنسدلة.
- أدخل سنة الصنع بين 1990 و 2025.
- أدخل عدد الكيلومترات المقطوعة. تأكد من أن الرقم معقول.
- اختر نوع الوقود من الخيارات المتاحة (البنزين، الديزل، إلخ).
- اختر نوع البائع بين فردي أو تاجر.
- اختر نوع ناقل الحركة (يدوي أو أوتوماتيكي).
- اختر عدد المالكين السابقين.
- أدخل سعة المحرك (بـ "سي سي").
- أدخل أقصى قوة للمحرك (بـ "حصان").
- أدخل عزم الدوران (بـ "نيوتن متر").
""")

# رسائل التنبيه في حال كانت المدخلات غير صحيحة أو ناقصة
if not (year and km_driven and fuel and seller_type and transmission and owner and mileage and engine and max_power and torque and seats and brand):
    st.warning("⚠️ تأكد من إدخال جميع الحقول بشكل صحيح.")

# إدخال المستخدم كـ DataFrame
input_data = pd.DataFrame({
    "year": [year],
    "km_driven": [km_driven],
    "fuel": [fuel],
    "seller_type": [seller_type],
    "transmission": [transmission],
    "owner": [owner],
    "mileage": [mileage],
    "engine": [engine],
    "max_power": [max_power],
    "torque": [torque],
    "seats": [seats],
    "brand": [brand]
})

st.write("🔢 المدخلات التي اخترتها:")
st.write(input_data)

# تحميل البيانات الأصلية
df_car = pd.read_csv(r"C:\Users\omar\Desktop\reg_cars_streamlit\data\reg_cars_selling.csv")
df_car = df_car.dropna()

# تنظيف وتحويل الأعمدة الرقمية النصية
df_car['mileage'] = df_car['mileage'].str.replace(r' kmpl$', '', regex=True).str.replace(r' km/kg$', '', regex=True).astype(float)
df_car['max_power'] = df_car['max_power'].str.replace(r' bhp$', '', regex=True).astype(float)
df_car['engine'] = df_car['engine'].str.replace(r' CC$', '', regex=True).astype(float)

# حفظ عمود العلامة التجارية
df_car['brand'] = df_car['name'].apply(lambda x: str(x).split()[0])
df_car['torque'] = df_car['torque'].astype(str)

# معالجة عمود العزم torque
def extract_torque(value):
    if pd.isna(value): return None
    value = str(value).lower().strip()
    matches = re.findall(r'([\d\.]+)\s*(nm|kgm)?', value, re.IGNORECASE)
    torque_values = []
    for match in matches:
        torque_value = float(match[0])
        unit = match[1].lower() if match[1] else 'nm'
        if unit == "kgm":
            torque_value *= 9.80665
        torque_values.append(torque_value)
    if torque_values:
        return sum(torque_values) / len(torque_values)
    return None

df_car['torque'] = df_car['torque'].apply(extract_torque)




# ترميز الأعمدة النصية
df_encoded = df_car.copy()
for col in df_encoded.select_dtypes(include='object').columns:
    df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])

# نموذج التدريب
X = df_encoded.drop(['selling_price', 'name'], axis=1)
y = df_encoded['selling_price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)

# إعداد بيانات الإدخال بنفس الترتيب والترميز
input_data_encoded = input_data.copy()
for col in ['fuel', 'seller_type', 'transmission', 'owner', 'brand']:
    encoder = LabelEncoder()
    encoder.fit(df_car[col])  # استخدم نفس البيانات الأصلية
    input_data_encoded[col] = encoder.transform(input_data_encoded[col])

# التنبؤ بالسعر
predicted_price = model.predict(input_data_encoded)[0]

# عرض السعر المتوقع
st.title("💰 السعر المتوقع بناءً على المدخلات:")
st.subheader(f"₹ {predicted_price:,.0f}")





# تحسين الرسم البياني وتنسيق الألوان
st.subheader("📊 مقارنة أسعار السيارات حسب نوع الوقود")

# نعرض المتوسطات لكل نوع وقود
fuel_avg_prices = df_car.groupby("fuel")["selling_price"].mean().sort_values(ascending=False)

st.write("### 💡 متوسط الأسعار حسب نوع الوقود:")
st.dataframe(fuel_avg_prices.reset_index().rename(columns={"fuel": "نوع الوقود", "selling_price": "متوسط السعر"}))

# رسم Bar Plot مع ألوان فخمة
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=fuel_avg_prices.index, y=fuel_avg_prices.values, palette="viridis", ax=ax)  # استخدام اللون الفخم
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)

# عرض الرسم البياني
st.pyplot(fig)



# streamlit run "C:\Users\omar\Desktop\reg_cars_streamlit\reg_cars1_project1_dashboard.py"