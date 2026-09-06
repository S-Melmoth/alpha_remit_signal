FROM golang:1.26-alpine AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -trimpath -o /behaviour ./cmd/behaviour && \
    CGO_ENABLED=0 go build -trimpath -o /pusher ./cmd/pusher && \
    CGO_ENABLED=0 go build -trimpath -o /payment-scheduler ./cmd/payment-scheduler

FROM alpine:3.22
RUN apk add --no-cache ca-certificates tzdata
COPY --from=build /behaviour /usr/local/bin/behaviour
COPY --from=build /pusher /usr/local/bin/pusher
COPY --from=build /payment-scheduler /usr/local/bin/payment-scheduler
USER 65532:65532
ENTRYPOINT ["behaviour"]
