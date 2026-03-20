import tarfile
import pymarc
import glob
import os, tarfile, re, json
from pymarc import parse_xml_to_array, MARCReader
import string
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity


class MARCXMLEmbeddingParser:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def parse_scsb_update_files(self, input_dir="data_marcxml", batch_size=10000):
        gz_files = self._find_gz_files(input_dir)
        self._extract_gz_files(gz_files, input_dir)
        xml_files = self._find_xml_files(input_dir)
        for file_idx, xml_file in enumerate(xml_files):
            # print(f"Parsing {xml_file}")
            marc_records = parse_xml_to_array(xml_file)
            # print(f"Parsed {len(marc_records)} records from {xml_file}")
            records = self._process_records(marc_records)
            self._save_batches(records, file_idx, xml_file, batch_size)

    def _find_gz_files(self, input_dir):
        gz_files = sorted(glob.glob(os.path.join(input_dir, "scsb_update_*.xml.gz")))
        print(f"Found {len(gz_files)} scsb_update_*.xml.gz files in {input_dir}")
        return gz_files

    def _extract_gz_files(self, gz_files, input_dir):
        os.makedirs(os.path.join(input_dir, "extracted"), exist_ok=True)
        import gzip

        for gz_path in gz_files:
            print(f"Processing {gz_path}")
            xml_filename = os.path.basename(gz_path).replace(".xml.gz", ".xml")
            extracted_xml_path = os.path.join(input_dir, "extracted", xml_filename)
            with gzip.open(gz_path, "rb") as gz_file:
                with open(extracted_xml_path, "wb") as xml_out:
                    xml_out.write(gz_file.read())

    def _find_xml_files(self, input_dir):
        xml_dir = os.path.join(input_dir, "extracted")
        xml_files = sorted(
            [
                os.path.join(xml_dir, f)
                for f in os.listdir(xml_dir)
                if f.endswith(".xml")
            ]
        )
        return xml_files

    def _process_records(self, marc_records):
        records = []
        for idx, record in enumerate(marc_records):
            # print(f"Record {idx}: type={type(record)}, content={record}")
            if record is None:
                # print(f"Record {idx} is None, skipping.")
                continue
            self.record = record
            # print(f"Processing record: {record}")
            record_id = self.id()
            title = self.title()
            transliterated_title = self.transliterated_title()
            publication_year = self.publication_year()
            pagination = self.pagination()
            edition = self.edition()
            context_title_index = self.context_title_index()
            publisher_name = self.publisher_name()
            type_of = self.type_of()
            title_part = self.title_part()
            title_number = self.title_number()
            author = self.author()
            title_inclusive_dates = self.title_inclusive_dates()
            text = " ".join(
                [
                    str(title),
                    str(title_part),
                    str(title_number),
                    str(title_inclusive_dates),
                    str(transliterated_title),
                    str(author),
                    str(publication_year),
                    str(pagination),
                    str(edition),
                    str(context_title_index),
                    str(publisher_name),
                    str(type_of),
                ]
            )
            print(f"Combined text for embedding: {text}")
            embedding = self.model.encode(text)
            records.append({"id": record_id, "text_embedding": embedding.tolist()})
        return records

    def _save_batches(self, records, file_idx, xml_file, batch_size):
        batches = [
            records[i : i + batch_size] for i in range(0, len(records), batch_size)
        ]
        for batch_idx, batch in enumerate(batches):
            print(
                f"Processing batch {batch_idx + 1} with {len(batch)} records from {xml_file}..."
            )
            os.makedirs("data_with_embeddings", exist_ok=True)
            batch_json_path = f"data_with_embeddings/scsb_update_{file_idx + 1}_batch_{batch_idx + 1}.json"
            with open(batch_json_path, "w") as f:
                json.dump(batch, f, indent=2)
            print(f"Saved batch embedding JSON: {batch_json_path}")

    def id(self):
        try:
            return self.record.get("001").data
        except KeyError, AttributeError:
            return ""

    def title(self):
        if self.__vernacular_title_field():
            title_field = self.__vernacular_title_field()
        else:
            title_field = self.__title_from_245()
        try:
            subfield_a = str(title_field.get("a") or "")
            subfield_b = str(title_field.get("b") or "")
            subfield_p = str(title_field.get("p") or "")
            title = " ".join([subfield_a, subfield_b, subfield_p])
            title = self.__strip_ending_punctuation(title)
            return title
        except KeyError, AttributeError:
            return ""

    def transliterated_title(self):
        title_field = self.__title_from_245()
        try:
            subfield_a = str(title_field.get("a") or "")
            subfield_b = str(title_field.get("b") or "")
            subfield_p = str(title_field.get("p") or "")
            title = " ".join([subfield_a, subfield_b, subfield_p])
            title = self.__strip_ending_punctuation(title)
            return title
        except KeyError, AttributeError:
            return ""

    def __title_from_245(self):
        try:
            if self.record is None:
                return ""
            title = self.record.get("245")
            if title is None:
                return ""
            return title
        except KeyError, TypeError, AttributeError:
            return ""

    def __vernacular_title_field(self):
        try:
            field_245 = self.record.get("245")
            if field_245 is None:
                return ""
            linked_fields = self.record.get_linked_fields(field_245)
            if linked_fields and len(linked_fields) > 0:
                return linked_fields[0]
            return ""
        except (
            KeyError,
            IndexError,
            AttributeError,
            pymarc.exceptions.MissingLinkedFields,
        ):
            return ""

    def publication_year(self):
        pub_year = None
        if self.date_one() and self.date_two():
            pub_year = self.date_two()
        elif self.date_one() and not self.date_two():
            pub_year = self.date_one()
        elif (
            not self.date_one() and not self.date_two() and self.__date_of_production()
        ):
            pub_year = self.__date_of_production()
        elif (
            not self.date_one()
            and not self.date_two()
            and not self.__date_of_production()
            and self.__date_of_publication()
        ):
            pub_year = self.__date_of_publication()
        return pub_year

    def pagination(self):
        try:
            subfield_a = self.record["300"].get("a")
            if subfield_a:
                return self.__normalize_extent(subfield_a)
            return ""
        except KeyError:
            return ""

    def context_title_index(self):
        try:
            return self.record["505"].get("t")
        except KeyError:
            return ""

    def edition(self):
        try:
            return self.__normalize_edition(self.record["250"].get("a"))
        except KeyError:
            return ""

    def publisher_name(self):
        try:
            pub = self.record["264"]["b"]
        except KeyError:
            try:
                pub = self.record["260"]["b"]
            except KeyError:
                return ""
        return self.__strip_punctuation(pub)

    def type_of(self):
        return self.record.leader.type_of_record

    def title_part(self):
        try:
            parts = self.record["245"].get_subfields("p")[1:]
            return self.__strip_punctuation(" ".join(parts))
        except KeyError:
            return ""

    def title_number(self):
        try:
            num = self.record["245"].get("n")
            if num:
                return self.__strip_punctuation(num)
            return ""
        except KeyError:
            return ""

    def author(self):
        if self.__vernacular_author_field():
            author_field = self.__vernacular_author_field()
        else:
            author_field = self.__author_from_1xx()

        if author_field:
            try:
                return self.__strip_ending_punctuation(author_field.get("a"))
            except AttributeError:
                return ""
        return ""

    def __author_from_1xx(self):
        try:
            return self.record["100"]
        except KeyError:
            try:
                return self.record["110"]
            except KeyError:
                try:
                    return self.record["111"]
                except KeyError:
                    return ""

    def __vernacular_author_field(self):
        try:
            return self.record.get_linked_fields(self.record["100"])[0]
        except KeyError, IndexError, pymarc.exceptions.MissingLinkedFields:
            try:
                return self.record.get_linked_fields(self.record["110"])[0]
            except KeyError, IndexError, pymarc.exceptions.MissingLinkedFields:
                try:
                    return self.record.get_linked_fields(self.record["111"])[0]
                except KeyError, IndexError, pymarc.exceptions.MissingLinkedFields:
                    return ""

    def title_inclusive_dates(self):
        try:
            date = self.record["245"].get("f")
            if date:
                return self.__strip_ending_punctuation(date)
            return ""
        except KeyError:
            return ""

    def __normalize_edition(self, edition):
        edition_mapping = {r"Ed\.": "Edition", r"ed\.": "edition"}
        try:
            for key, value in edition_mapping.items():
                edition = re.sub(key, value, edition)
            return self.__strip_punctuation(edition)
        except TypeError:
            return ""

    def __normalize_extent(self, extent):
        extent_mapping = {
            r"p\.": "pages",
            r"v\.": "volumes",
            r"vol\.": "volumes",
            r"ℓ\.": "leaves",
        }
        for key, value in extent_mapping.items():
            extent = re.sub(key, value, extent)
        return self.__strip_punctuation(extent)

    def __strip_ending_punctuation(self, some_string):
        punctuation_to_strip = string.punctuation.replace(")", "")
        return some_string.strip(punctuation_to_strip + " ")

    def __strip_punctuation(self, some_string):
        punctuation_to_strip = string.punctuation.replace("&", "")
        some_string = some_string.translate(str.maketrans("", "", punctuation_to_strip))
        some_string = re.sub("  ", " ", some_string).strip()
        return some_string

    def is_valid_date(self, date_string):
        valid = True
        if date_string == "9999":
            valid = False
        elif date_string == "    ":
            valid = False
        elif self.number_of_characters(date_string) != 4:
            valid = False
        try:
            int(date_string)
        except ValueError, TypeError:
            valid = False
        return valid

    def number_of_characters(self, date_string):
        try:
            return len(date_string)
        except TypeError:
            return False

    def date_one(self):
        try:
            date_string = self.record["008"].data[7:11]
            return self.__as_date(date_string)
        except KeyError:
            return None

    def date_two(self):
        try:
            date_string = self.record["008"].data[11:15]
            return self.__as_date(date_string)
        except KeyError:
            return None

    def __date_of_production(self):
        try:
            date_string = self.record["264"]["c"]
        except KeyError:
            return None
        return self.__as_date(date_string)

    def __date_of_publication(self):
        try:
            date_string = self.record["260"]["c"]
        except KeyError:
            return None
        return self.__as_date(date_string)

    def __as_date(self, date_string):
        # Remove punctuation (for 260 and 264 fields)
        date_string = self.__strip_punctuation(date_string)
        if self.is_valid_date(date_string):
            return int(date_string)
        return ""

    marcxml_dir = "data_marcxml"

    # def extract_and_parse_marcxml(self, marcxml_dir, batch_size=10000):
    #     print(f"Looking for tar.gz files in {marcxml_dir}...")
    #     tar_files = sorted(
    #         glob.glob(os.path.join(marcxml_dir, "incremental_*_new.tar.gz"))
    #     )
    #     print(f"Found {len(tar_files)} tar.gz files in {marcxml_dir}")
    #     for tar_idx, tar_gz_path in enumerate(tar_files):
    #         print(f"Processing {tar_gz_path}")
    #         extract_dir = "data_marcxml/extracted"
    #         os.makedirs(extract_dir, exist_ok=True)
    #         with tarfile.open(tar_gz_path, "r:gz") as tar:
    #             tar.extractall(path=extract_dir)
    #         print(f"Extracted {tar_gz_path} to {extract_dir}")
    #         for f in os.listdir(extract_dir):
    #             file_path = os.path.join(extract_dir, f)
    #             if not f.endswith(".xml"):
    #                 new_file_path = file_path + ".xml"
    #                 os.rename(file_path, new_file_path)
    #         marc_files = [
    #             os.path.join(extract_dir, f)
    #             for f in os.listdir(extract_dir)
    #             if f.endswith(".xml")
    #         ]
    #         print(f"Found {len(marc_files)} MARC files.")
    #         records = []
    #         records = parse_xml_to_array(
    #             "data_marcxml/extracted/incremental_46370036360006421_20260305_020345[024]_new.xml"
    #         )
    #         print(f"Number of records: {len(records)}")
    #         print(type(records[0]) if records else "No records")
    #         records = []

    #         for marc_file in marc_files:
    #             marc_records = parse_xml_to_array(marc_file)
    #             print(f"Parsed {len(marc_records)} records from {marc_file}")
    #             for idx, record in enumerate(marc_records):
    #                 print(f"Record {idx}: type={type(record)}, content={record}")
    #                 if record is None:
    #                     print(f"Record {idx} is None, skipping.")
    #                     continue
    #                 self.record = record
    #                 print(f"Processing record: {record}")
    #                 record_id = self.id()
    #                 # Join field values and calculate one text embedding
    #                 title = self.title()
    #                 transliterated_title = self.transliterated_title()
    #                 publication_year = self.publication_year()
    #                 pagination = self.pagination()
    #                 edition = self.edition()
    #                 context_title_index = self.context_title_index()
    #                 publisher_name = self.publisher_name()
    #                 type_of = self.type_of()
    #                 title_part = self.title_part()
    #                 title_number = self.title_number()
    #                 author = self.author()
    #                 title_inclusive_dates = self.title_inclusive_dates()
    #                 text = " ".join(
    #                     [
    #                         str(title),
    #                         str(title_part),
    #                         str(title_number),
    #                         str(title_inclusive_dates),
    #                         str(transliterated_title),
    #                         str(author),
    #                         str(publication_year),
    #                         str(pagination),
    #                         str(edition),
    #                         str(context_title_index),
    #                         str(publisher_name),
    #                         str(type_of),
    #                     ]
    #                 )
    #                 print(f"Combined text for embedding: {text}")
    #                 embedding = self.model.encode(text)
    #                 records.append(
    #                     {"id": record_id, "text_embedding": embedding.tolist()}
    #                 )
    #             # Batching logic: split records into batches of batch_size
    #             batches = [
    #                 records[i : i + batch_size]
    #                 for i in range(0, len(records), batch_size)
    #             ]
    #             for batch_idx, batch in enumerate(batches):
    #                 print(
    #                     f"Processing batch {batch_idx + 1} with {len(batch)} records from {marc_file}..."
    #                 )
    #                 os.makedirs("data_with_embeddings", exist_ok=True)
    #                 batch_json_path = f"data_with_embeddings/marcxml_embeddings_{tar_idx + 1}_batch_{batch_idx + 1}.json"
    #                 with open(batch_json_path, "w") as f:
    #                     json.dump(batch, f, indent=2)
    #                 print(f"Saved batch embedding JSON: {batch_json_path}")

    def create_embedding_matrix(self):
        """
        Loads all marcxml_embeddings_*_batch_1.json files from data_with_embeddings/,
        creates a matrix of embeddings (IDs as rows, embedding dims as columns),
        and saves the result as embeddings_matrix/scsb_update_*_matrix.csv.
        """
        embedding_files = sorted(
            glob.glob("data_with_embeddings/scsb_update_*_batch_*.json")
        )
        os.makedirs("embeddings_matrix", exist_ok=True)
        for batch_idx, file in enumerate(embedding_files):
            all_ids = []
            all_embeddings = []
            with open(file, "r") as f:
                data = json.load(f)
            for item in data:
                all_ids.append(item["id"])
                all_embeddings.append(item["text_embedding"])
            if not all_embeddings:
                print(f"No embeddings found in {file}.")
                continue
            num_dims = len(all_embeddings[0])
            columns = [f"dim_{i + 1}" for i in range(num_dims)]
            df = pd.DataFrame(all_embeddings, columns=columns)
            output_path = (
                f"embeddings_matrix/scsb_update_batch_{batch_idx + 1}_matrix.csv"
            )
            df.to_csv(output_path, header=True, index=False)
            print(f"Saved embedding matrix (with column names, no IDs): {output_path}")
        return None


if __name__ == "__main__":
    parser = MARCXMLEmbeddingParser()
    marcxml_dir = "data_marcxml"
    # parser.extract_and_parse_marcxml(marcxml_dir)
    # parser.create_embedding_matrix()
    # parser.parse_scsb_update_files(input_dir=marcxml_dir, batch_size=10000)
