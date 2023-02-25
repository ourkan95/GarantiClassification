import os
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

USER_ID = 'user_id' 
TARGET = 'moved_after_2019'

df_skills = pd.read_csv(r'C:\Users\ourka\OneDrive\Masaüstü\garanti\garanti-bbva-data-camp\skills.csv')
df_education = pd.read_csv(r'C:\Users\ourka\OneDrive\Masaüstü\garanti\garanti-bbva-data-camp\education.csv')
df_languages = pd.read_csv(r'C:\Users\ourka\OneDrive\Masaüstü\garanti\garanti-bbva-data-camp\languages.csv')
df_work_experiences = pd.read_csv(r'C:\Users\ourka\OneDrive\Masaüstü\garanti\garanti-bbva-data-camp\work_experiences.csv')

#LANGUAGE
df_languages = df_languages[df_languages['language'].notnull() & df_languages['proficiency'].notnull()]

def encode_proficiency(value):
    if value == 'elementary':
        return 1
    elif value == 'limited_working':
        return 2
    elif value == 'professional_working':
        return 3
    elif value == 'full_professional':
        return 4
    elif value == 'native_or_bilingual':
        return 5
    else:
        return 0
    
df_languages['proficiency'] = df_languages['proficiency'].apply(encode_proficiency)

df_languages.language.value_counts().head(30)

df_languages.loc[df_languages['language'].str.contains("Eng|eng|İng|ing|Ing|English|english|İngilizce|ingilizce|Ingilizce|İNGİLİZCE|INGILIZCE"), 'language'] = "English"
df_languages.loc[df_languages['language'].str.contains("Tür|Tur|tür|tur|TURKISH|TÜRKÇE|Tükçe"), 'language'] = "Turkish"
df_languages.loc[df_languages['language'].str.contains("Alm|alm|ger|Ger|Deu|deu|ALMANCA|GER|DEU|ALM"), 'language'] = "German"
df_languages.loc[df_languages['language'].str.contains("Fr|fr|FR"), 'language'] = "French"
df_languages.loc[df_languages['language'].str.contains("Spa|spa|İsp|isp|SPA|ISP|İSP"), 'language'] = "Spanish"
df_languages.loc[df_languages['language'].str.contains("Rus|rus|RUS"), 'language'] = "Russian"
df_languages.loc[df_languages['language'].str.contains("ara|Ara|ARA"), 'language'] = "Arabic" 
df_languages.loc[df_languages['language'].str.contains("Chinese|CHINESE|CHI|Chi|chi|Çince"), 'language'] = "Chinese"
df_languages.loc[df_languages['language'].str.contains("Japanese|Japonca|japa|japo|JAPA|JAPO"), 'language'] = "Japanese"
df_languages.loc[df_languages['language'].str.contains("İtalyanca|Italian|itali|ital|ITA|İTAL"), 'language'] = "Italian"
df_languages.loc[df_languages['language'].str.contains("Azerice|Azerbaijani|Azeri|Azərba"), 'language'] = "Azerbaijani" 

df_languages.loc[~df_languages["language"].isin(["English","Turkish","German","French","Spanish","Russian","Arabic","Chinese","Japanese","Italian","Azerbaijani"]), "language"] = "OtherLanguage"


df_languages = df_languages.drop_duplicates(['user_id', 'language'])
df_languages = pd.pivot(df_languages, index='user_id', columns='language', values='proficiency')
df_languages = df_languages.fillna(0).astype(int)
df_languages.head()

df_languages.to_csv('language.csv')

#EDUCATION

df_education = df_education[df_education['school_name'].notnull() & df_education['degree'].notnull()]

df_education.loc[df_education['degree'].str.contains("Associate|Ön|ön lisans|önlisans", na=False), 'degree'] = "associate degree"
df_education.loc[df_education['degree'].str.contains("Doctor|Doktor|Ph", na=False), 'degree'] = "phd"
df_education.loc[df_education['degree'].str.contains("Yüksek|Master|MSc|MS|M.Sc.|MBA|Msc|M.Sc|M.S.|M.S", na=False), 'degree'] = "master's degree"
df_education.loc[df_education['degree'].str.contains("BS|Bs|Bachelor|BSc|BE|B.E.|B.Sc.|B.S.|B.S|Engineer|BA|BBA|BEng|B.B.A.|B.A.Sc.|Undergraduate|Licentiate|Licence|License|Lisans|lisans|Bacheleer|bachelor|Bsc", na=False), 'degree'] = "bachelor degree"

df_education.loc[~df_education["degree"].isin(["associate degree","phd","master's degree","bachelor degree"]), "degree"] = "others"

"""
1) Dünya sıralamasında ilk 20 de yer alan üniversiteler : TheFirstLevel(World) // https://www.timeshighereducation.com/world-university-rankings/2022/world-ranking#!/page/0/length/25/sort_by/rank/sort_order/asc/cols/stats
2) Dünya sıralamasında ilk 1000 de yer alan TÜRK üniversiteler : TheFirstLevel(TR) // https://tr.euronews.com/2022/10/12/dunyanin-en-iyi-universiteleri-aciklandi-ilk-500de-turkiyeden-sadece-1-universite-var
3) TR sıralamasında ilk 30 da yer alan üniversiteler : TheSecondLevel(TR) // https://egezegen.com/egitim/turkiyenin-en-iyi-universiteleri-siralamasi/
4) TR sıralamasında ilk 50 de yer alan üniversiteler : TheLowLevel(TR) // https://egezegen.com/egitim/turkiyenin-en-iyi-universiteleri-siralamasi/
"""

df_education.loc[df_education['school_name'].str.contains("Oxford|Kaliforniya|Harvard|Stanford|Cambridge|Massachusetts|Princeton|Kaliforniya| Berkeley|Yale|Chicago|Kolombiya|Imperial|Johns Hopkins|Pensilvanya|ETH Zürih|Pekin|Tsinghua|Toronto|Londra", na=False), 'school_name'] = "top20world"
df_education.loc[df_education['school_name'].str.contains("Çankaya|Koç|Sabancı|ODTÜ|Bahçeşehir|Hacettepe|İstanbul Teknik|Istanbul Technical University|Bilkent|Boğaziçi|Düzce|Fırat|İstanbul Medeniyet|Özyeğin|Cankaya|Koc|Sabanci|Odtu|Bahcesehir|Istanbul Teknik|Bogazici|Düzce|Firat|Istanbul Medeniyet|Ozyegin", na=False), 'school_name'] = "top1000tr"
df_education.loc[df_education['school_name'].str.contains("İstanbul Üniversitesi|Istanbul University|Ankara|Ege|İhsan Doğramacı|Bilkent|Gebze|Gazi|Yıldız|Yildiz Technical University|Sabancı|İzmir Yüksek Teknoloji|Atatürk|Bezm-i Alem|Erciyes|Marmara|Dokuz Eylül|Selçuk|Çukurova|Karadeniz Teknik|Eskişehir Osmangazi|Akdeniz|Abdullah Gül|Bursa Uludağ|Ondokuz Mayıs|İnönü|Anadolu", na=False), 'school_name'] = "top30tr"
df_education.loc[df_education['school_name'].str.contains("Süleyman Demirel|Gaziantep|Sakarya|Çankaya|Kocaeli|Van Yüzüncü|İzmir Katip Çelebi|Yıldırım Beyazıt|Başkent|Atılım|Dicle|Manisa Celâl Bayar|Pamukkale|Tobb Ekonomi Ve Teknoloji|Acıbadem Mehmet Ali Aydınlar|Mersin|Yeditepe", na=False), 'school_name'] = "top50tr"

df_education.loc[~df_education['school_name'].str.contains("top20world|top1000tr|top30tr|top50tr"), 'school_name' ] = "others"



df_education = df_education.drop_duplicates(['user_id', 'degree'])
df_education = pd.pivot(df_education, index='user_id', columns='degree', values='school_name')
df_education.head()

df_education.to_csv('education.csv')

#EXP
df_temp = pd.DataFrame()
df_work_experiences = df_work_experiences[df_work_experiences['start_year_month'] < 201901]
df_work_experiences = df_work_experiences.sort_values(by=['user_id', 'start_year_month'])

df_temp['company(1th)'] = df_work_experiences.groupby(USER_ID)['company_id'].nth(-1).astype(str)
df_temp['company(2th)'] = df_work_experiences.groupby(USER_ID)['company_id'].nth(-2).astype(str)
df_temp['company(3th)'] = df_work_experiences.groupby(USER_ID)['company_id'].nth(-3).astype(str)
df_temp['company(4th)'] = df_work_experiences.groupby(USER_ID)['company_id'].nth(-4).astype(str)
df_temp['company(5th)'] = df_work_experiences.groupby(USER_ID)['company_id'].nth(-5).astype(str)


df_temp['min_exp_time'] = df_work_experiences.groupby(USER_ID)['start_year_month'].min()
df_temp['max_exp_time'] = df_work_experiences.groupby(USER_ID)['start_year_month'].max()

df_temp['company_count_2018'] = df_work_experiences[df_work_experiences['start_year_month'].gt(201712)].groupby(USER_ID).size()
df_temp['company_count_2017'] = df_work_experiences[df_work_experiences['start_year_month'].gt(201612)].groupby(USER_ID).size()
df_temp['company_count_2016'] = df_work_experiences[df_work_experiences['start_year_month'].gt(201512)].groupby(USER_ID).size()

df_work_experiences = df_temp
df_work_experiences.head()

df_work_experiences.to_csv('exp.csv')

#SKILLS

df_skills.loc[df_skills['skill'].str.contains("HTML|CSS|JavaScript|Bootstrap|jQuery|AngularJS|React.js|Angular|JSP|AJAX|Front-end|frontend|Vue.js"), 'skill'] = "FRONTEND"
df_skills.loc[df_skills['skill'].str.contains("ASP|.NET|PHP|php|Java|java|Node.js|\#|Go|Spring Boot|Eclipse|backend"), 'skill'] = "BACKEND"
df_skills.loc[df_skills['skill'].str.contains("SQL|Database|MongoDB|Postgre|PL/|MySQL|Oracle|Veritabanı|Hibernate|Veritabanları|Big Data|veritabanı|Mongo|sql"), 'skill'] = "DATABASE"
df_skills.loc[df_skills['skill'].str.contains("OOP|Object|Nesne"), 'skill'] = "OOP"
df_skills.loc[df_skills['skill'].str.contains("Teamwork|Ekip Çalışması|İletişim|Team Motivation"), 'skill'] = "TEAMWORKER"
df_skills.loc[df_skills['skill'].str.contains("Yazılım Geliştirme|Software Development|Jenkins|Software Design|Maven|UML|TFS|JIRA|DevOps|Yazılım|Design patterns"), 'skill'] = "SOFTWARE_DEVELOPMENT"
df_skills.loc[df_skills['skill'].str.contains("Agile|AGILE"), 'skill'] = "AGILE"
df_skills.loc[df_skills['skill'].str.contains("Excel|Office|Word|PowerPoint|office"), 'skill'] = "MICROSOFT_OFFICE"
df_skills.loc[df_skills['skill'].str.contains("WEB|Web"), 'skill'] = "WEB"
df_skills.loc[df_skills['skill'].str.contains("Management|Yönetim|Lider|Leadership|Proje planlama"), 'skill'] = "MANAGEMENT"
df_skills.loc[df_skills['skill'].str.contains("Machine Learning|Veri Bilimi|Veri Analizi|Algorithms|Analysis|Makine Öğrenimi|Algoritmalar|Yapay Zeka|Artificial|Doğal Dil İşleme|İstatistik|Neural Networks|machine learning"), 'skill'] = "ML"
df_skills.loc[df_skills['skill'].str.contains("Programlama|Programming"), 'skill'] = "PROGRAMMING"
df_skills.loc[df_skills['skill'].str.contains("Android|mobil|MOBILE|Mobile|Mobil Uygulamalar|Flutter|React Native|Kotlin"), 'skill'] = "ANDROID"
df_skills.loc[df_skills['skill'].str.contains("Framework|WCF|Django"), 'skill'] = "FRAMEWORK"
df_skills.loc[df_skills['skill'].str.contains("Unity|GAME|game"), 'skill'] = "GAME_DEV"
df_skills.loc[df_skills['skill'].str.contains("Araştırma|Research"), 'skill'] = "RESEARCH"
df_skills.loc[df_skills['skill'].str.contains("Mühendislik|Engineering"), 'skill'] = "ENGINEERING"
df_skills.loc[df_skills['skill'].str.contains("Embedded Systems|Microservices|AutoCAD|Arduino|SolidWorks|hardware|Donanım"), 'skill'] = "HARDWARE_SKILLS"
df_skills.loc[df_skills['skill'].str.contains("Problem Solving|Sorun Çözme"), 'skill'] = "PROBLEM_SOLVING"
df_skills.loc[df_skills['skill'].str.contains("Server|Tomcat|Docker|Redis|sunucu"), 'skill'] = "SERVER"
df_skills.loc[df_skills['skill'].str.contains("CLOUD|Cloud Computing|bulut|Kubernetes"), 'skill'] = "CLOUD"
df_skills.loc[df_skills['skill'].str.contains("PYTHON|Python|py|Pandas|Numpy"), 'skill'] = "PYTHON"
df_skills.loc[df_skills['skill'].str.contains("C+|cplusplus|c+"), 'skill'] = "C++"
df_skills.loc[df_skills['skill'].str.contains("Git|github|GITHUB"), 'skill'] = "GITHUB"
df_skills.loc[df_skills['skill'].str.contains("Linux|LINUX"), 'skill'] = "LINUX"
df_skills.loc[df_skills['skill'].str.contains("Photoshop|Photo|Adobe|DESIGN|design|illustrator|3D|Grafik|Tasarım"), 'skill'] = "DESIGNER"
df_skills.loc[df_skills['skill'].str.contains("Networking|network"), 'skill'] = "NETWORKING"
df_skills.loc[df_skills['skill'].str.contains("ECONOMY|PAYMENT|Payment|economy|ekonomi"), 'skill'] = "ECONOMY"
df_skills.loc[df_skills['skill'].str.contains("FINANCE|Finance|Finans|satış|finans|pazarlama|Sales"), 'skill'] = "FINANCE"
df_skills.loc[df_skills['skill'].str.contains("Testing|test|TEST|Manuel Test Etme|JUnit"), 'skill'] = "TEST"
df_skills.loc[df_skills['skill'].str.contains("Siber Güvenlik|Siber|cybersecurity|security|5C|Güvenliği"), 'skill'] = "CYBERSECURITY"
df_skills.loc[df_skills['skill'].str.contains("iOS|ios|IOS|Swift"), 'skill'] = "IOS"
df_skills.loc[df_skills['skill'].str.contains("Marketing"), 'skill'] = "MARKETING"

df_skills.loc[~df_skills["skill"].isin(["FRONTEND","BACKEND","DATABASE","OOP","TEAMWORKER","SOFTWARE_DEVELOPMENT","AGILE", 
                                       "MICROSOFT_OFFICE","WEB","MANAGEMENT","ML","PROGRAMMING","ANDROID","FRAMEWORK","GAME_DEV", 
                                       "RESEARCH","ENGINEERING","HARDWARE_SKILLS","PROBLEM_SOLVING","SERVER","CLOUD","PYTHON","C++", 
                                       "GITHUB","LINUX","DESIGNER","NETWORKING","ECONOMY","FINANCE","TEST",
                                       "CYBERSECURITY","IOS","MARKETING"]), "skill"] = "OtherSkill"


df_skills['have'] = True
df_skills = df_skills.drop_duplicates(['user_id', 'skill'])
df_skills = pd.pivot(df_skills, index='user_id', columns='skill', values='have')
df_skills = df_skills.fillna(0).astype(int)

df_skills.to_csv('skills.csv')





























