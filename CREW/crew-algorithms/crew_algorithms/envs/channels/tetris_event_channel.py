from cattrs.preconf.json import JsonConverter
from mlagents_envs.side_channel import IncomingMessage

from crew_algorithms.envs.channels.game_event_channel import GameEventChannel
from crew_algorithms.envs.channels.messages.tetris import (
    NoMessage,
    ObjectSpawnedEventMessage,
    converter,
)


class TetrisEventChannel(GameEventChannel[ObjectSpawnedEventMessage | NoMessage]):
    @property
    def converter(self) -> JsonConverter:
        return converter

    def decode_message(
        self, msg: IncomingMessage
    ) -> NoMessage | ObjectSpawnedEventMessage:
        return self.converter.loads(
            msg.read_string(),
            ObjectSpawnedEventMessage | NoMessage,
        )
