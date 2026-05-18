"""
[ROLE]
Point d'entrée principal du projet gaitpdb.

[RESPONSIBILITY]
- Lancer la validation robuste du modèle parcimonieux (Phase 3b).
- Servir de wrapper minimal vers le code métier situé dans project/.

[INPUTS]
- Aucun (paramètres par défaut via project.config).

[OUTPUTS]
- Résultats de validation dans le dossier output/.

[ASSUMPTIONS]
- Le dataset est présent dans le dossier datasets/.

[RISKS]
- Aucun.

[DEPENDENCIES]
- project.validate
"""

from __future__ import annotations

from project.validate import main

if __name__ == "__main__":
    main()
