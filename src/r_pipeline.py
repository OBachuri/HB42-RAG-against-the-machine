
from pathlib import Path
import sys
from pydantic import RootModel
# import json
from tqdm import tqdm


from src.r_bm25 import r_bm25_retrieve, r_bm25_load
from src.r_semantic import RSentenceTransformer

from src.r_data_model import RetrieveMode, RagDataset, StudentSearchResults
from src.r_data_model import RetrievedChunk, MinimalSearchResults
from src.r_data_model import MinimalSource

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.__main__ import RagCLI


class RPipeLine():
    """ Class to work with Hybrid retrieving """

    @classmethod
    def retrive(cls,
                param: RagCLI,
                query: str = "",
                print_chunks: bool = False,
                rs: RSentenceTransformer | None = None
                ) -> list[RetrievedChunk]:

        """ Retrive chucks for one question """

        RRF_influence_bm25 = 1
        RRF_influence_vector = 0.8
        RRF_bm25_min_scope = 0.01
        RRF_k = 60

        k = param.k
        res = []
        print_ = print_chunks
        if param.retrieve_mode == RetrieveMode.HYBRID:
            if k < 30:
                param.k += 15
            if print_ and not (param.print_debug):
                print_ = False

        if (param.retrieve_mode == RetrieveMode.BM25
           or param.retrieve_mode == RetrieveMode.HYBRID):

            if (param._retriver is None) or (not param._chunks):
                param._retriver, param._chunks = r_bm25_load(param)

            res = r_bm25_retrieve(
                param=param,
                retriever=param._retriver,
                chunks=param._chunks,
                query=query,
                print_chunks=print_)
            if print_chunks:
                print("-"*30)

        if (param.retrieve_mode == RetrieveMode.EMBEDDINGS
           or param.retrieve_mode == RetrieveMode.HYBRID):
            if rs is None:
                with RSentenceTransformer(param) as rs_:
                    if rs_:
                        res.extend(
                            rs_.retrive(
                                param,
                                query=query,
                                print_chunks=print_))
            else:
                res.extend(rs.retrive(param,
                                      query=query,
                                      print_chunks=print_))

            if print_chunks:
                print("-"*30)

        if param.retrieve_mode == RetrieveMode.HYBRID:
            if (k != param.k):
                param.k = k
            if print_chunks:
                print("Hybrid retrieved:")
            rrf_scores: dict[str, float] = {}
            i = 1
            metod = None
            for r in res:
                if (metod is None) or (metod != r.metod):
                    i = 1
                    metod = r.metod
                if metod == RetrieveMode.BM25:
                    if r.score > RRF_bm25_min_scope:
                        score = RRF_influence_bm25/(RRF_k + i)
                    else:
                        score = RRF_influence_bm25/(RRF_k + i + k)
                else:
                    score = RRF_influence_vector/(RRF_k + i)
                rrf_scores[r.id] = rrf_scores.get(r.id, 0) + score
                # print(i, r.id, score, metod)
                i += 1
            ids = [k_ for k_, _ in sorted(rrf_scores.items(),
                                          key=lambda item: -item[1])][:k]
            res_dic = {v.id: v for v in res if v.id in ids}
            res = [res_dic[i] for i in ids]
            if print_chunks:
                for i_ in ids:
                    c_ = res_dic[i_]
                    score = rrf_scores[i_]
                    if param.print_debug:
                        print(c_.file_path,
                              f"[{c_.first_character_index}:"
                              f"{c_.last_character_index}]"
                              f" (score: {score:.3f}) id={c_.id}")
                    else:
                        print(c_.file_path,
                              f"[{c_.first_character_index}:"
                              f"{c_.last_character_index}]"
                              f" (score: {score:.3f})")

        return res

    @classmethod
    def retrieve_dataset(cls,
                         param: RagCLI,
                         write_file: bool = True
                         ) -> list[MinimalSearchResults]:

        """ Retrive chucks for questions from dataset """

        # Read file with query
        file_path = Path(str(param.dataset_path))
        if not (file_path.is_file() and file_path.stat().st_size > 0):
            print("Error: Can't read query set. No file or empty file. "
                  f"({file_path})\n",
                  file=sys.stderr)
            sys.exit(1)

        try:
            # Read the raw text from the file
            with open(file_path, "r", encoding="utf-8") as q_file:
                json_string = q_file.read()
            # Parse and validate the JSON string back into Pydantic objects
            queries = RootModel[RagDataset].model_validate_json(
                json_string).root
        except Exception as ex:
            print("Error: Can't read query from file"
                  f"({file_path})\n",
                  ex,
                  file=sys.stderr)
            sys.exit(1)

        file_name = file_path.name

        print("Loaded", len(queries.rag_questions), "questions")

        search_result: list[MinimalSearchResults] = []

        rs = RSentenceTransformer(param)

        for q in tqdm(queries.rag_questions,
                      desc="Search chunks for Question",
                      unit="Question"):
            chunks_fond = cls.retrive(
                param=param,
                query=q.question,
                print_chunks=False,
                rs=rs)

            minimal_sources = [
                MinimalSource.model_validate(chunk.model_dump())
                for chunk in chunks_fond
                ]

            search_result.append(MinimalSearchResults(
                question_id=q.question_id,
                question=q.question,
                question_str=q.question,
                retrieved_sources=minimal_sources))
            # print(q.question_id)
            # print(chunks_fond)

        rs.close()

        if write_file:
            st_result = StudentSearchResults(k=param.k,
                                             search_results=search_result)

            json_string = RootModel(st_result).model_dump_json(indent=2)
            file_path = Path(param.save_directory) / file_name

            try:
                # This creates the folders if they do not exist
                file_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception as ex:
                print("Error: can't create folder to store"
                      "dataset_docs_public.json! \n",
                      ex, file=sys.stderr)
                sys.exit(1)

            try:
                # Writing to JSON file
                with open(file_path, "w", encoding="utf-8") as res_file:
                    res_file.write(json_string)
                print("  Saved student_search_results to :", file_path)
            except Exception as ex:
                print("Error: can't store student_search_results!"
                      f"({file_path})\n",
                      ex, file=sys.stderr)
                sys.exit(1)

        return (search_result)
