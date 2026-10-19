"""Valida células em ordem no processo IPython quando kernel TCP é indisponível.

Para uso interativo, prefira Restart Kernel and Run All no Jupyter/VS Code.
"""

from pathlib import Path
from io import BytesIO
import os
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
from IPython.display import display, Image
import matplotlib.pyplot as plt


def execute(path):
    path = Path(path).resolve()
    os.chdir(path.parent.parent)
    notebook = nbformat.read(path, as_version=4)
    shell = InteractiveShell.instance()

    def show(*args, **kwargs):
        for number in plt.get_fignums():
            buffer = BytesIO()
            plt.figure(number).savefig(
                buffer, format="png", dpi=110, bbox_inches="tight"
            )
            display(Image(data=buffer.getvalue()))
        plt.close("all")

    original_show = plt.show
    plt.show = show
    try:
        count = 0
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            count += 1
            with capture_output() as captured:
                result = shell.run_cell(cell.source, store_history=True)
            cell.outputs = []
            cell.execution_count = count
            for name, value in [
                ("stdout", captured.stdout),
                ("stderr", captured.stderr),
            ]:
                if value:
                    cell.outputs.append(
                        nbformat.v4.new_output("stream", name=name, text=value)
                    )
            for out in captured.outputs:
                cell.outputs.append(
                    nbformat.v4.new_output(
                        "display_data", data=out.data, metadata=out.metadata
                    )
                )
            nbformat.write(notebook, path)
            if result.error_before_exec or result.error_in_exec:
                raise RuntimeError(
                    str(result.error_before_exec or result.error_in_exec)
                )
            print(f"Célula {count}: concluída", flush=True)
    finally:
        plt.show = original_show


if __name__ == "__main__":
    execute(
        Path(__file__).resolve().parents[1] / "notebooks/Iris_Generativa_NatGenAI.ipynb"
    )
