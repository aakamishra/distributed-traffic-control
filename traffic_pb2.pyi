from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class DataReply(_message.Message):
    __slots__ = ["error_code", "time", "version"]
    ERROR_CODE_FIELD_NUMBER: _ClassVar[int]
    TIME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    error_code: str
    time: int
    version: int
    def __init__(self, version: _Optional[int] = ..., time: _Optional[int] = ..., error_code: _Optional[str] = ...) -> None: ...

class DataTransmission(_message.Message):
    __slots__ = ["api_key", "entries", "intersection_id", "timestamp", "version"]
    class KeyValue(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: int
        def __init__(self, key: _Optional[str] = ..., value: _Optional[int] = ...) -> None: ...
    API_KEY_FIELD_NUMBER: _ClassVar[int]
    ENTRIES_FIELD_NUMBER: _ClassVar[int]
    INTERSECTION_ID_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    api_key: str
    entries: _containers.RepeatedCompositeFieldContainer[DataTransmission.KeyValue]
    intersection_id: str
    timestamp: str
    version: int
    def __init__(self, version: _Optional[int] = ..., intersection_id: _Optional[str] = ..., timestamp: _Optional[str] = ..., entries: _Optional[_Iterable[_Union[DataTransmission.KeyValue, _Mapping]]] = ..., api_key: _Optional[str] = ...) -> None: ...

class RefreshReply(_message.Message):
    __slots__ = ["error_code", "time", "version"]
    ERROR_CODE_FIELD_NUMBER: _ClassVar[int]
    TIME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    error_code: str
    time: int
    version: int
    def __init__(self, version: _Optional[int] = ..., time: _Optional[int] = ..., error_code: _Optional[str] = ...) -> None: ...

class RefreshRequest(_message.Message):
    __slots__ = ["api_key", "intersection_id", "phase_id", "time", "version"]
    API_KEY_FIELD_NUMBER: _ClassVar[int]
    INTERSECTION_ID_FIELD_NUMBER: _ClassVar[int]
    PHASE_ID_FIELD_NUMBER: _ClassVar[int]
    TIME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    api_key: str
    intersection_id: str
    phase_id: int
    time: int
    version: int
    def __init__(self, version: _Optional[int] = ..., time: _Optional[int] = ..., phase_id: _Optional[int] = ..., intersection_id: _Optional[str] = ..., api_key: _Optional[str] = ...) -> None: ...

class StateReply(_message.Message):
    __slots__ = ["error_code", "version"]
    ERROR_CODE_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    error_code: str
    version: int
    def __init__(self, version: _Optional[int] = ..., error_code: _Optional[str] = ...) -> None: ...

class StateRequest(_message.Message):
    __slots__ = ["message", "version"]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    message: str
    version: int
    def __init__(self, version: _Optional[int] = ..., message: _Optional[str] = ...) -> None: ...
