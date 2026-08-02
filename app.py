import streamlit as st

from ChatService import ChatService
from DocumentParser import DocumentParser, DocumentParserError
from LocalLLMClient import (
    MODEL_CATALOG,
    LocalLLMClient,
    LocalLLMError,
)


st.set_page_config(
    page_title="ZtQ - Sprachmodell",
    page_icon="🧠",
    layout="centered",
)


DEFAULT_MODEL_LABEL = "Qwen2.5 1.5B – empfohlen"
CUSTOM_MODEL_LABEL = "Eigenes Hugging-Face-Modell"


@st.cache_resource(show_spinner=False)
def load_llm(
    model_name: str,
    max_new_tokens: int,
    trust_remote_code: bool,
) -> LocalLLMClient:
    """Lädt genau das aktuell ausgewählte Modell."""

    return LocalLLMClient(
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        trust_remote_code=trust_remote_code,
    )


def initialize_session_state() -> None:
    """Legt die benötigten Sitzungsvariablen an."""

    default_model = MODEL_CATALOG[DEFAULT_MODEL_LABEL]

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "active_model_selector" not in st.session_state:
        st.session_state.active_model_selector = (
            DEFAULT_MODEL_LABEL
        )

    if "active_model_name" not in st.session_state:
        st.session_state.active_model_name = (
            default_model.model_id
        )

    if "active_max_new_tokens" not in st.session_state:
        st.session_state.active_max_new_tokens = (
            default_model.max_new_tokens
        )

    if "active_trust_remote_code" not in st.session_state:
        st.session_state.active_trust_remote_code = (
            default_model.trust_remote_code
        )


initialize_session_state()


st.title("🧠 ZtQ - Sprachmodell")

st.caption(
    "Lokaler, modellagnostischer Assistent für "
    "Textüberarbeitung und Dokumentfragen."
)


model_options = [
    *MODEL_CATALOG.keys(),
    CUSTOM_MODEL_LABEL,
]

active_selector = st.session_state.active_model_selector

if active_selector not in model_options:
    active_selector = DEFAULT_MODEL_LABEL


with st.sidebar:
    st.header("Sprachmodell")

    selected_model_label = st.selectbox(
        "Modell auswählen",
        options=model_options,
        index=model_options.index(active_selector),
    )

    if selected_model_label == CUSTOM_MODEL_LABEL:
        custom_model_name = st.text_input(
            "Hugging-Face-Modell-ID",
            value=(
                st.session_state.active_model_name
                if (
                    st.session_state.active_model_selector
                    == CUSTOM_MODEL_LABEL
                )
                else ""
            ),
            placeholder="Organisation/Modellname",
            help=(
                "Beispiel: Qwen/Qwen2.5-1.5B-Instruct"
            ),
        )

        custom_max_tokens = st.number_input(
            "Maximale Antwortlänge",
            min_value=50,
            max_value=2000,
            value=500,
            step=50,
        )

        custom_trust_remote_code = st.checkbox(
            "Externen Modellcode erlauben",
            value=False,
            help=(
                "Nur aktivieren, wenn das Modell dies ausdrücklich "
                "benötigt und die Quelle vertrauenswürdig ist."
            ),
        )

        selected_model_name = custom_model_name.strip()
        selected_max_tokens = int(custom_max_tokens)
        selected_trust_remote_code = (
            custom_trust_remote_code
        )

        st.caption(
            "Hier kann grundsätzlich jedes kompatible "
            "Causal-Language-Model eingetragen werden."
        )

    else:
        selected_definition = MODEL_CATALOG[
            selected_model_label
        ]

        selected_model_name = (
            selected_definition.model_id
        )

        selected_max_tokens = (
            selected_definition.max_new_tokens
        )

        selected_trust_remote_code = (
            selected_definition.trust_remote_code
        )

        st.caption(
            selected_definition.description
        )

        st.code(selected_model_name)

    load_button = st.button(
        "Ausgewähltes Modell laden",
        use_container_width=True,
        type="primary",
    )

    if load_button:
        if not selected_model_name:
            st.error(
                "Bitte zuerst eine Hugging-Face-Modell-ID eingeben."
            )

        else:
            st.session_state.active_model_selector = (
                selected_model_label
            )

            st.session_state.active_model_name = (
                selected_model_name
            )

            st.session_state.active_max_new_tokens = (
                selected_max_tokens
            )

            st.session_state.active_trust_remote_code = (
                selected_trust_remote_code
            )

            st.session_state.messages = []

            # Entfernt das bisherige Modell aus dem
            # Streamlit-Ressourcencache.
            load_llm.clear()

            st.rerun()


try:
    with st.spinner(
        "Sprachmodell wird geladen. Beim ersten Einsatz eines "
        "Modells kann der Download einige Minuten dauern …"
    ):
        llm_client = load_llm(
            model_name=st.session_state.active_model_name,
            max_new_tokens=(
                st.session_state.active_max_new_tokens
            ),
            trust_remote_code=(
                st.session_state.active_trust_remote_code
            ),
        )

    chat_service = ChatService(llm_client)
    document_parser = DocumentParser()

except LocalLLMError as exc:
    st.error(str(exc))
    st.stop()


with st.sidebar:
    st.divider()

    st.header("Arbeitsmodus")

    mode = st.radio(
        "Funktion auswählen",
        options=[
            "Text überarbeiten",
            "Dokumente befragen",
        ],
    )

    st.divider()

    uploaded_files = st.file_uploader(
        "Dokumente hochladen",
        type=[
            "pdf",
            "docx",
            "txt",
            "md",
        ],
        accept_multiple_files=True,
        help=(
            "Dateien auf diese Fläche ziehen oder über "
            "die Dateiauswahl öffnen."
        ),
    )

    document_parts: list[str] = []
    readable_files = 0

    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                extracted_text = document_parser.parse(
                    uploaded_file=uploaded_file,
                    filename=uploaded_file.name,
                )

                document_parts.append(
                    f"===== DATEI: {uploaded_file.name} =====\n\n"
                    f"{extracted_text}"
                )

                readable_files += 1

                st.success(
                    f"{uploaded_file.name} eingelesen"
                )

            except DocumentParserError as exc:
                st.warning(str(exc))

    document_context = "\n\n".join(
        document_parts
    )

    if uploaded_files:
        st.info(
            f"{readable_files} von "
            f"{len(uploaded_files)} Dateien lesbar."
        )

    st.divider()

    st.subheader("Aktives Modell")

    st.code(
        st.session_state.active_model_name
    )

    st.write("Ausführung über:")

    st.code(
        llm_client.get_device_name()
    )

    if st.button(
        "Chatverlauf löschen",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()


if not st.session_state.messages:
    if mode == "Text überarbeiten":
        st.info(
            "Gib unten einen Text ein. Das ausgewählte "
            "Sprachmodell erstellt eine überarbeitete Fassung."
        )

    else:
        st.info(
            "Lade links ein oder mehrere Dokumente hoch und "
            "stelle anschließend eine Frage dazu."
        )


for message in st.session_state.messages:
    with st.chat_message(
        message["role"]
    ):
        st.markdown(
            message["content"]
        )


if mode == "Dokumente befragen":
    input_placeholder = (
        "Frage zu den hochgeladenen Dokumenten …"
    )

    input_disabled = not bool(
        document_context
    )

else:
    input_placeholder = (
        "Text zur Überarbeitung eingeben …"
    )

    input_disabled = False


user_input = st.chat_input(
    input_placeholder,
    disabled=input_disabled,
)


if user_input:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner(
            "Antwort wird erstellt …"
        ):
            try:
                if mode == "Dokumente befragen":
                    answer = (
                        chat_service.answer_from_documents(
                            question=user_input,
                            document_context=document_context,
                        )
                    )

                else:
                    answer = (
                        chat_service.smooth_text(
                            user_input
                        )
                    )

                st.markdown(answer)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

            except (
                LocalLLMError,
                ValueError,
            ) as exc:
                st.error(str(exc))