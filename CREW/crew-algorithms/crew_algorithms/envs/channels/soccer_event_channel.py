from cattrs.preconf.json import JsonConverter
from mlagents_envs.side_channel import IncomingMessage

from crew_algorithms.envs.channels.game_event_channel import GameEventChannel
from crew_algorithms.envs.channels.messages.soccer import (
    GameEndedEventMessage,
    GameScoredEventMessage,
    GameStartedEventMessage,
    converter,
)


class SoccerEventChannel(
    GameEventChannel[
        GameStartedEventMessage | GameEndedEventMessage | GameScoredEventMessage
    ]
):
    @property
    def converter(self) -> JsonConverter:
        return converter

    def decode_message(
        self, msg: IncomingMessage
    ) -> GameStartedEventMessage | GameEndedEventMessage | GameScoredEventMessage:
        return self.converter.loads(
            msg.read_string(),
            GameStartedEventMessage | GameEndedEventMessage | GameScoredEventMessage,
        )
