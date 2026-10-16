from typing import List, Optional
from datetime import datetime
from storage.models import RetrievalResult
from storage.db import DatabaseManager


class TemporalValidityFilter:
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or DatabaseManager()

    def filter_and_annotate(
        self,
        results: List[RetrievalResult],
        reference_date: Optional[str] = None,
        filter_out_invalid: bool = False,
    ) -> List[RetrievalResult]:
        if not reference_date:
            return results

        try:
            ref_dt = datetime.strptime(reference_date.strip(), "%Y-%m-%d")
        except Exception:
            return results

        annotated: List[RetrievalResult] = []

        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            for res in results:
                eff_start = res.effective_start
                eff_end = res.effective_end

                # Query version history if exists
                cursor.execute(
                    "SELECT * FROM provision_versions WHERE provision_id = ? ORDER BY version_num ASC",
                    (res.provision_id,)
                )
                versions = [dict(r) for r in cursor.fetchall()]

                status = "VALID"
                if not eff_start:
                    status = "UNDETERMINED"
                else:
                    try:
                        start_dt = datetime.strptime(eff_start, "%Y-%m-%d")
                        if ref_dt < start_dt:
                            status = "INVALID_NOT_YET_EFFECTIVE"
                        elif eff_end:
                            end_dt = datetime.strptime(eff_end, "%Y-%m-%d")
                            if ref_dt > end_dt:
                                status = "INVALID_EXPIRED"
                    except Exception:
                        status = "UNDETERMINED"

                # Check amendment versions
                if versions and status == "VALID":
                    v1 = versions[0]
                    v2 = versions[1] if len(versions) > 1 else None

                    if v2 and v2.get("effective_start"):
                        try:
                            v2_start_dt = datetime.strptime(v2["effective_start"], "%Y-%m-%d")
                            if ref_dt < v2_start_dt:
                                status = "VALID_PRIOR_TO_AMENDMENT"
                                res.evidence_text = (res.evidence_text or "") + f" [Lưu ý: Tại ngày tham chiếu {reference_date}, điều khoản này có hiệu lực nguyên bản (v1) trước khi được sửa đổi ngày {v2['effective_start']}]."
                        except Exception:
                            pass

                res.validity_status = status

                if filter_out_invalid:
                    if status.startswith("VALID") or status == "UNDETERMINED":
                        annotated.append(res)
                else:
                    annotated.append(res)

        return annotated
