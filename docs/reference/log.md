# Logging

::: cartpole.log
        handler: python
        options:
            filters: [
                '!to_ns', '!to_stamp', '!this_or_now', '!pylog_level', '!get_pylogger'
            ]


