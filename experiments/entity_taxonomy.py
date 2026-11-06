"""Verify the public role PDF pin and export project role paraphrases."""
import hashlib
import json
import urllib.request
from pathlib import Path
from pypdf import PdfReader
from semeval.entity import ROLE_SOURCE,ROLE_SHA256,taxonomy,HYPOTHESES
from semeval.data import ROLE_GROUPS

def main():
    path=Path('data/public/official/ENTITY-ROLE-TAXONOMY.pdf');path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():path.write_bytes(urllib.request.urlopen(ROLE_SOURCE,timeout=45).read())
    if hashlib.sha256(path.read_bytes()).hexdigest()!=ROLE_SHA256:raise ValueError('Organizer role PDF changed; verify before updating the pin')
    text=PdfReader(path).pages[0].extract_text()
    roles=taxonomy()
    compact=''.join(text.split())  # PDF kerning splits initials, e.g. F oreign.
    if len(roles)!=22 or any(role.replace(' ','') not in compact for role in roles):raise ValueError('Official role inventory differs')
    value=dict(source=ROLE_SOURCE,source_sha256=ROLE_SHA256,counts=dict(coarse=3,fine=22),groups=ROLE_GROUPS,
        labels=[dict(id=key,**row,hypothesis=HYPOTHESES[key]) for key,row in roles.items()],
        definition_source='Concise project paraphrases and working hypotheses; no organizer examples.',submission_scorer_parity=False)
    for name in ('semeval/taxonomies/entity-roles.json','website/dist/entity-roles.json'):
        Path(name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Verified 22 fine roles / 3 parents against the pinned organizer PDF.')

if __name__=='__main__':main()
