from vectorDB import LawVectorDB

LAW_NAME_MAPPING = {
    "고용보험법": "employment_insurance",
    "근로기준법": "labor_standards",
    "남녀고용평등법": "gender_equality_employment"
}

POLICY_NAME_MAPPING = {
    "K-육아정책.pdf": "parenting_policy",
    "청년내일채움공제.pdf": "youth_policy"
}

law_data_dir = "./data/law"
policy_data_dir = "./data/policy"

vector_db = LawVectorDB()
vector_db.bulk_ingest_from_mapping(LAW_NAME_MAPPING, law_data_dir)
vector_db.bulk_ingest_policies(POLICY_NAME_MAPPING, policy_data_dir)